"""Input-adaptive matmul reduction for the TADA Llama backbone.

Adapted from "Reduced Matrix Multiplication: Input-Adaptive Matrix-Product
Reduction for LLM Inference" (arXiv:2608.13426). RMM is a training-free,
inference-time method: for each linear in the Transformer it selects the most
informative slices along the contraction (input-channel) dimension from the
current activations and contracts only those, leaving the model weights, the
``inputs_embeds -> last_hidden_state`` contract, and all output shapes
untouched. A single retention ratio (1.0 = exact) dials the accuracy-efficiency
trade-off.

Adapted components (see the PR description for the full accounting):
- The paper's custom A100 kernels are replaced with ``index_select`` plus
  ``F.linear`` over the retained slices. FLOPs drop by the retained fraction;
  wall-clock gains beyond that need the custom kernels.
- The paper's importance criterion is implemented as a parameter-free
  activation-magnitude times weight-column-magnitude proxy.
- The paper finds attention-side matmuls substantially more reducible than MLP
  ones, so ``components="attention"`` (q/k/v/o projections) is the default.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

Components = str  # "attention" | "mlp" | "all"

_PROJECTION_SUFFIX = "_proj"


@dataclass
class MatmulStats:
    """Per-module counters accumulated across reduced forward calls."""

    calls: int = 0
    flops_full: float = 0.0
    flops_reduced: float = 0.0
    elements_seen: int = 0

    @property
    def flops_fraction(self) -> float:
        """FLOPs actually spent, as a fraction of the unreduced product."""
        if self.flops_full <= 0.0:
            return 1.0
        return self.flops_reduced / self.flops_full


class ReducedMatmulLinear(nn.Linear):
    """``nn.Linear`` that contracts only the top-retention input channels.

    The wrapper keeps the weight and bias of the linear it replaces (shared
    storage, same state-dict keys), so checkpoints, ``.to()``/dtype casts, and
    module paths are unchanged. At ``retention >= 1.0`` it is an exact
    pass-through of the original product.
    """

    def __init__(self, linear: nn.Linear, retention: float = 1.0):
        super().__init__(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        self.weight = linear.weight
        if linear.bias is not None:
            self.bias = linear.bias
        self.retention = float(retention)
        self.stats = MatmulStats()
        # Kept unregistered so the wrapper adds no state-dict keys and the
        # original module survives for an exact restore.
        object.__setattr__(self, "_orig", linear)
        object.__setattr__(self, "_col_importance", None)

    @property
    def retained_features(self) -> int:
        return max(1, min(self.in_features, int(round(self.in_features * self.retention))))

    def _column_importance(self) -> torch.Tensor:
        cached = self._col_importance
        if cached is None or cached.device != self.weight.device or cached.dtype != self.weight.dtype:
            cached = self.weight.detach().abs().sum(dim=0)
            object.__setattr__(self, "_col_importance", cached)
        return cached

    def _select(self, x: torch.Tensor) -> torch.Tensor:
        """Pick the contraction indices with the largest contribution magnitude."""
        flat = x.detach().reshape(-1, self.in_features)
        score = flat.abs().sum(dim=0).to(self._column_importance().dtype) * self._column_importance()
        keep = min(self.retained_features, self.in_features)
        return torch.topk(score, keep).indices.sort().values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.numel() // self.in_features
        self.stats.calls += 1
        self.stats.elements_seen += rows
        self.stats.flops_full += 2.0 * rows * self.in_features * self.out_features

        if self.retention >= 1.0 or self.in_features <= 1:
            return F.linear(x, self.weight, self.bias)

        idx = self._select(x)
        self.stats.flops_reduced += 2.0 * rows * idx.numel() * self.out_features
        return F.linear(x.index_select(-1, idx), self.weight.index_select(1, idx), self.bias)


def _component_of(path: str) -> str | None:
    parts = path.split(".")
    if "self_attn" in parts:
        return "attention"
    if "mlp" in parts:
        return "mlp"
    return None


def _targets(component: Components, path: str, child_name: str) -> bool:
    if not child_name.endswith(_PROJECTION_SUFFIX):
        return False
    kind = _component_of(path)
    return component == "all" or kind == component


def apply_reduced_matmul(
    model: nn.Module,
    retention: float = 0.75,
    components: Components = "attention",
) -> list[str]:
    """Wrap the backbone's projection linears with :class:`ReducedMatmulLinear`.

    Args:
        model: The Transformer backbone (e.g. ``TadaForCausalLM.model``).
        retention: Fraction of contraction-dim slices kept, in ``(0, 1]``.
            ``1.0`` keeps the exact product while still recording stats.
        components: ``"attention"`` (q/k/v/o projections, default — the paper
            finds these the most reducible), ``"mlp"``, or ``"all"``.

    Returns:
        The replaced module paths.
    """
    if not 0.0 < retention <= 1.0:
        raise ValueError(f"retention must be in (0, 1], got {retention}")
    if components not in ("attention", "mlp", "all"):
        raise ValueError(f"components must be 'attention', 'mlp' or 'all', got {components!r}")

    replaced: list[str] = []
    for path, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, ReducedMatmulLinear) or not isinstance(child, nn.Linear):
                continue
            full = f"{path}.{child_name}" if path else child_name
            if not _targets(components, full, child_name):
                continue
            setattr(module, child_name, ReducedMatmulLinear(child, retention=retention))
            replaced.append(full)
    return replaced


def restore_full_matmul(model: nn.Module) -> list[str]:
    """Undo :func:`apply_reduced_matmul`, restoring the original linears."""
    restored: list[str] = []
    for path, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, ReducedMatmulLinear):
                continue
            setattr(module, child_name, child._orig)
            restored.append(f"{path}.{child_name}" if path else child_name)
    return restored


def reduced_matmul_stats(model: nn.Module) -> dict[str, dict]:
    """Aggregate per-module RMM counters plus model-level totals."""
    per_module: dict[str, dict] = {}
    totals = MatmulStats()
    for path, module in model.named_modules():
        if not isinstance(module, ReducedMatmulLinear):
            continue
        stats: MatmulStats = module.stats
        totals.calls += stats.calls
        totals.flops_full += stats.flops_full
        totals.flops_reduced += stats.flops_reduced
        per_module[path] = {
            "retention": module.retention,
            "kept_features": module.retained_features,
            "calls": stats.calls,
            "flops_fraction": stats.flops_fraction,
        }
    per_module["total"] = {
        "calls": totals.calls,
        "flops_full": totals.flops_full,
        "flops_reduced": totals.flops_reduced,
        "flops_fraction": totals.flops_fraction,
    }
    return per_module


__all__ = [
    "Components",
    "MatmulStats",
    "ReducedMatmulLinear",
    "apply_reduced_matmul",
    "reduced_matmul_stats",
    "restore_full_matmul",
]
