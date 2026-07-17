"""Weight-only quantization with signed-gradient (SignSGD) rounding.

Adapted from *Optimize Weight Rounding via Signed Gradient Descent for the
Quantization of LLMs* (SignRound, arXiv:2309.05516). The paper's core
mechanism is implemented at full fidelity: a per-weight rounding variable
``V`` and an optional per-group weight-clipping factor ``beta`` are optimized
with **SignSGD** (signed gradient descent, fixed learning rate, no momentum)
to minimize the **output reconstruction error** of the quantized weight
matrix. This is the part of the paper that delivers its accuracy win.

Two auxiliary components of the paper are substituted with target-native
equivalents so the method slots onto TADA's Llama-backbone loading path
without new infrastructure:

* **Transformer-block reconstruction** (the paper optimizes one Transformer
  block at a time against that block's frozen input activations, requiring a
  calibration corpus and a block-forward harness) is replaced by
  **per-Linear reconstruction** against a representative sample of the
  Linear's own input activations. The SignSGD-over-rounding core is identical;
  only the reconstruction granularity is coarsened from "block" to "layer",
  which needs no calibration corpus and no block harness.
* **Packed-int4 deployment kernels** (GPTQ/AutoGPTQ-style CUDA/AutoGPTQ
  integration) are out of scope: results are stored as ``int8`` weights plus a
  per-group ``float`` scale (a real ~4x parameter-memory reduction for the
  quantized layers) and dequantized inside :class:`QuantizedLinear.forward`,
  so the rest of the model is untouched. Wiring a fused int4 kernel is the
  natural follow-up.
"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "QuantizedLinear",
    "optimize_linear",
    "optimize_model_weight_rounding",
    "optimize_weight_rounding",
]


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest in the forward pass, identity in the backward pass (straight-through)."""
    return (x.round() - x).detach() + x


def _reshape_for_groups(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    """Pad ``weight`` (out, in) along ``in`` to a multiple of ``group_size``; return (padded, pad)."""
    in_features = weight.shape[1]
    pad = (-in_features) % group_size
    if pad:
        weight = F.pad(weight, (0, pad))
    return weight, pad


def _expand_scale(scale: torch.Tensor, group_size: int, out_features: int) -> torch.Tensor:
    """Broadcast a per-group scale (out, n_groups) back out to (out, n_groups * group_size)."""
    n_groups = scale.shape[1]
    return scale.unsqueeze(-1).expand(-1, -1, group_size).reshape(out_features, n_groups * group_size)


def optimize_weight_rounding(
    weight: torch.Tensor,
    sample_inputs: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    iters: int = 200,
    lr: float = 0.01,
    optimize_clipping: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SignRound-optimize the rounding of a single weight matrix.

    Args:
        weight: ``(out_features, in_features)`` float weight matrix to quantize.
        sample_inputs: ``(N, in_features)`` representative input activations used to
            build the output-reconstruction objective.
        bits: target weight bit-width (symmetric signed; ``qmax = 2**(bits-1) - 1``).
        group_size: symmetric grouping along ``in_features``; clamped to ``<= in_features``.
        iters: SignSGD iterations over the rounding variable ``V`` (and ``beta``).
        lr: SignSGD step size. The paper uses ~0.01.
        optimize_clipping: also optimize a per-group clipping factor ``beta``.

    Returns:
        ``(q_int, scale, dequant)`` where ``q_int`` is the int8 quantized weight
        (out, in), ``scale`` is the per-group float scale (out, n_groups), and
        ``dequant`` is the dequantized weight ``q_int * scale`` (out, in).
    """
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in [2, 8], got {bits}")
    out_features, in_features = weight.shape
    group_size = max(1, min(group_size, in_features))
    qmax = 2 ** (bits - 1) - 1

    w = weight.detach().to(torch.float32)
    wp, pad = _reshape_for_groups(w, group_size)  # (out, in+pad)
    wg = wp.view(out_features, -1, group_size)
    base_scale = (wg.abs().amax(dim=-1) / qmax).clamp_min(1e-8)  # (out, n_groups)

    x = sample_inputs.detach().to(torch.float32)
    xp = F.pad(x, (0, pad)) if pad else x
    target = xp @ wp.T  # (N, out) reference Linear output

    v = torch.zeros_like(wp, requires_grad=True)  # rounding nudge, init 0 => round-to-nearest
    beta = torch.ones_like(base_scale, requires_grad=True)  # per-group clipping factor, init 1.0

    best_loss: float | None = None
    best_v = v.detach().clone()
    best_beta = beta.detach().clone()

    for _ in range(iters + 1):  # +1 so the round-to-nearest init is evaluated as a candidate
        scale = beta * base_scale
        scale_exp = _expand_scale(scale, group_size, out_features)
        q = _ste_round(wp / scale_exp + v).clamp(-qmax, qmax)
        wq = q * scale_exp
        loss = F.mse_loss(xp @ wq.T, target)
        grad_v, grad_beta = torch.autograd.grad(loss, (v, beta))

        with torch.no_grad():
            loss_val = loss.item()
            if best_loss is None or loss_val < best_loss:
                best_loss = loss_val
                best_v = v.clone()
                best_beta = beta.clone()
            v = (v - lr * grad_v.sign()).clamp(-0.5, 0.5)
            if optimize_clipping:
                beta = (beta - lr * grad_beta.sign()).clamp_min(0.0)
            v.requires_grad_(True)
            beta.requires_grad_(True)

    with torch.no_grad():
        scale = best_beta * base_scale
        scale_exp = _expand_scale(scale, group_size, out_features)
        q = torch.round(wp / scale_exp + best_v).clamp(-qmax, qmax)[:, :in_features]
        dequant = q * scale_exp[:, :in_features]
        q_int = q.to(torch.int8)

    return q_int, scale.detach(), dequant.detach()


class QuantizedLinear(nn.Module):
    """An ``nn.Linear`` whose weights are stored as ``int8`` + a per-group ``float`` scale.

    Parameter memory is ~4x smaller than the equivalent float32 ``nn.Linear``; the
    weight is dequantized (``q_int * scale``) inside ``forward`` so the module is a
    drop-in replacement for ``nn.Linear`` on the forward path.
    """

    def __init__(
        self,
        q_int: torch.Tensor,
        scale: torch.Tensor,
        in_features: int,
        out_features: int,
        group_size: int,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer("q_int", q_int.to(torch.int8))
        self.register_buffer("scale", scale.to(torch.float32))
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

    def dequantized_weight(self) -> torch.Tensor:
        """Materialize the dequantized (float) weight (out_features, in_features)."""
        scale_exp = _expand_scale(self.scale, self.group_size, self.out_features)[:, : self.in_features]
        return self.q_int.to(self.scale.dtype) * scale_exp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized_weight().to(x.dtype)
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, bias={self.bias is not None}"
        )


def optimize_linear(
    layer: nn.Linear,
    sample_inputs: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    iters: int = 200,
    lr: float = 0.01,
    optimize_clipping: bool = True,
) -> QuantizedLinear:
    """SignRound-quantize an ``nn.Linear`` into a :class:`QuantizedLinear`."""
    out_features, in_features = layer.weight.shape
    group_size = max(1, min(group_size, in_features))
    q_int, scale, _ = optimize_weight_rounding(
        layer.weight.data,
        sample_inputs,
        bits=bits,
        group_size=group_size,
        iters=iters,
        lr=lr,
        optimize_clipping=optimize_clipping,
    )
    bias = getattr(layer, "bias", None)
    quantized = QuantizedLinear(q_int, scale, in_features, out_features, group_size, bias=bias)
    return quantized.to(layer.weight.device)


def _default_backbone_filter(name: str, module: nn.Module) -> bool:
    """Select the Llama backbone Linears (everything under the ``model.`` sub-module)."""
    return name.startswith("model.") and isinstance(module, nn.Linear)


def _get_parent(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def optimize_model_weight_rounding(
    model: nn.Module,
    bits: int = 4,
    calibration_ids: torch.Tensor | None = None,
    seq_len: int = 64,
    batch: int = 2,
    group_size: int = 128,
    iters: int = 200,
    lr: float = 0.01,
    optimize_clipping: bool = True,
    module_filter: Callable[[str, nn.Module], bool] | None = None,
) -> int:
    """Apply SignRound weight-only quantization to a model's backbone Linears, in place.

    One forward pass through the backbone captures each target Linear's input
    activations (the reconstruction objective's calibration signal); each target is
    then replaced in place by a :class:`QuantizedLinear`. Returns the number of
    Linears quantized.
    """
    if module_filter is None:
        module_filter = _default_backbone_filter
    backbone = getattr(model, "model", model)

    if calibration_ids is None:
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", 32000)
        calibration_ids = torch.randint(0, vocab_size, (batch, seq_len))
    calibration_ids = calibration_ids.to(next(backbone.parameters()).device)

    captures: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]):
            if name not in captures:
                inp = inputs[0]
                captures[name] = inp.detach().to(torch.float32).reshape(-1, inp.shape[-1])

        return hook

    targets: list[tuple[str, nn.Linear]] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name, module in model.named_modules():
        if module_filter(name, module):
            handles.append(module.register_forward_pre_hook(make_hook(name)))
            targets.append((name, module))

    try:
        with torch.no_grad():
            backbone(input_ids=calibration_ids)
    finally:
        for handle in handles:
            handle.remove()

    count = 0
    for name, module in targets:
        sample = captures.get(name)
        if sample is None or sample.shape[0] == 0:
            continue
        quantized = optimize_linear(
            module,
            sample,
            bits=bits,
            group_size=group_size,
            iters=iters,
            lr=lr,
            optimize_clipping=optimize_clipping,
        )
        parent, attr = _get_parent(model, name)
        setattr(parent, attr, quantized)
        count += 1
    return count
