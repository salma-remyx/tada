"""Post-training affine weight quantization for ``nn.Linear`` backbone layers.

Adapted from *AffineQuant: Affine Transformation Quantization for Large Language
Models* (You et al., 2024, https://arxiv.org/abs/2403.12544).

Core mechanism (kept at full fidelity)
---------------------------------------
Existing LLM post-training weight-quantization methods only optimize a *scaling*
transformation before quantizing. AffineQuant instead optimizes a full *affine*
(scaling + rotation) invertible matrix ``A`` per ``nn.Linear`` weight so that
quantizing ``A @ W`` reconstructs the layer's output as closely as possible.
Because ``A`` is a full matrix rather than a per-channel diagonal, it can rotate
outlier/structured weight directions onto the quantization grid, reducing output
error beyond what scaling alone achieves. The inverse ``A^{-1}`` is then folded
back so the layer's matmul forward contract is preserved:

    y = x @ W^T          (original)
    y ≈ (x @ Q(A @ W)^T) @ (A^{-1})^T

where ``Q`` is a per-output-channel symmetric round-to-nearest quantizer.

Adapted (Mode 2) substitutions — what is target-native, not from the paper
-------------------------------------------------------------------------
* **Optimizer**: AffineQuant's bespoke steepest-descent-with-partial-gradient
  optimizer is replaced by the repo-native ``torch.optim.Adam`` path.
* **Calibration data**: the paper's 128-sample C4/PTB activation corpus
  (captured via forward hooks on real inputs) is replaced by a parameter-free
  synthetic Gaussian calibration activation generated in-place per layer. This
  keeps the capability usable out-of-the-box with no external data or model
  forward; pass real captured activations to ``optimize_affine_transform`` for a
  faithful fit.
* **Compensation folding**: the paper folds ``A^{-1}`` into the *next* layer's
  input path (so no extra op survives at inference). Here each layer carries a
  self-contained output-side ``@ (A^{-1})^T`` compensation (see
  :class:`AffineQuantizedLinear`). This keeps every layer independent and
  verifiable; folding into the next layer is a deployment optimization left to a
  follow-up.
* **Feasibility constraint**: the paper bounds ``||A^{-1}||_1`` to keep the
  inverse well-conditioned. We approximate this with a light L2 pull of ``A``
  toward identity plus an identity initialization, which empirically keeps
  ``A`` invertible. Layers wider than ``max_affine_dim`` fall back to plain RTN
  (``A = I``) because the full ``out x out`` optimization becomes
  memory/compute-bound; this matches AffineQuant's own per-layer cost profile.

The flow-matching / diffusion head is intentionally *not* touched — only the
Llama backbone ``nn.Linear`` projections are quantized.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "AffineQuantizedLinear",
    "optimize_affine_transform",
    "quantize_weight",
    "apply_affine_quantization",
]


def quantize_weight(weight: torch.Tensor, bits: int, *, straight_through: bool = False) -> torch.Tensor:
    """Per-output-channel symmetric round-to-nearest quantization.

    Args:
        weight: ``[out_features, in_features]`` weight matrix.
        bits: target bit-width (>=2). Symmetric grid in ``[-(2^(b-1)-1), 2^(b-1)-1]``.
        straight_through: if True, the rounding is a straight-through estimator
            (forward rounds, backward passes the gradient through unchanged) so the
            affine transform can be optimized end-to-end through the quantizer.

    Returns:
        Dequantized weight on the quantization grid, same shape as ``weight``.
    """
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")
    q_max = 2 ** (bits - 1) - 1
    # Per output-channel (row) symmetric scale.
    max_abs = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = max_abs / q_max
    scaled = (weight / scale).round().clamp(-q_max, q_max) * scale
    if straight_through:
        return weight + (scaled - weight).detach()
    return scaled


def _reconstruction_error(
    weight: torch.Tensor,
    affine: torch.Tensor,
    calibration_inputs: torch.Tensor,
    target: torch.Tensor,
    bits: int,
) -> float:
    """True post-quantization output reconstruction MSE for a given affine matrix ``A``.

    Uses hard (non-straight-through) quantization, so this is exactly the error an
    :class:`AffineQuantizedLinear` with this ``A`` would incur — the quantity the
    optimizer cares about.
    """
    with torch.no_grad():
        quantized = quantize_weight(affine @ weight, bits, straight_through=False)
        pred = (calibration_inputs @ quantized.t()) @ torch.linalg.inv(affine).t()
        return F.mse_loss(pred, target).item()


def optimize_affine_transform(
    weight: torch.Tensor,
    calibration_inputs: torch.Tensor,
    bits: int,
    *,
    num_steps: int = 25,
    learning_rate: float = 5e-3,
    regularization: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimize an invertible affine matrix ``A`` minimizing output reconstruction error.

    Minimizes ``|| X @ W^T - (X @ Q(A @ W)^T) @ (A^{-1})^T ||^2`` over ``A`` via
    Adam, initialized to identity so the starting point equals plain RTN. Returns
    the hard-quantized transformed weight ``Q(A @ W)`` and the compensation matrix
    ``(A^{-1})^T`` to install in an :class:`AffineQuantizedLinear`.

    A best-checkpoint safeguard tracks the lowest true reconstruction error seen
    across optimization, seeded with the identity (RTN) point. Because the
    straight-through estimator is exact in the forward pass, the tracked loss is
    the true post-quantization reconstruction error, so the returned transform is
    guaranteed to reconstruct the layer output no worse than plain RTN regardless
    of optimization noise.

    Args:
        weight: ``[out_features, in_features]`` original (float32) weight.
        calibration_inputs: ``[N, in_features]`` calibration activations.
        bits: quantization bit-width.
        num_steps: Adam steps over ``A``.
        learning_rate: Adam learning rate.
        regularization: L2 pull of ``A`` toward identity (stand-in for the
            paper's ``||A^{-1}||_1`` feasibility constraint).

    Returns:
        ``(quantized_weight, a_inv_t)`` where ``a_inv_t == (A^{-1})^T``.
    """
    out_features = weight.shape[0]
    eye = torch.eye(out_features, dtype=weight.dtype, device=weight.device)
    affine = eye.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([affine], lr=learning_rate)
    with torch.no_grad():
        target = calibration_inputs @ weight.t()  # original outputs X @ W^T, [N, out]
        best_affine = eye.clone()
        best_error = _reconstruction_error(weight, best_affine, calibration_inputs, target, bits)
    for _ in range(num_steps):
        optimizer.zero_grad()
        quantized = quantize_weight(affine @ weight, bits, straight_through=True)  # Q(A @ W)
        pred = (calibration_inputs @ quantized.t()) @ torch.linalg.inv(affine).t()  # [N, out]
        recon = F.mse_loss(pred, target)
        if regularization:
            loss = recon + regularization * ((affine - eye) ** 2).mean()
        else:
            loss = recon
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            error = _reconstruction_error(weight, affine, calibration_inputs, target, bits)
            if error < best_error:
                best_error = error
                best_affine = affine.detach().clone()
    with torch.no_grad():
        quantized_weight = quantize_weight(best_affine @ weight, bits, straight_through=False)
        a_inv_t = torch.linalg.inv(best_affine).t().contiguous()
    return quantized_weight.detach(), a_inv_t.detach()


class AffineQuantizedLinear(nn.Module):
    """An ``nn.Linear`` whose weight was affine-pre-transformed then quantized.

    Forward computes ``(x @ Wq^T) @ (A^{-1})^T (+ bias)`` to reconstruct the
    original layer output under quantization. Buffers store the quantized
    transformed weight ``Wq`` and the compensation ``(A^{-1})^T``.
    """

    def __init__(self, quantized_weight: torch.Tensor, a_inv_t: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        out_features, in_features = quantized_weight.shape
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", quantized_weight.detach().clone())
        self.register_buffer("a_inv_t", a_inv_t.detach().clone())
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight)
        y = y @ self.a_inv_t
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def _make_affine_quantized_linear(
    linear: nn.Linear,
    bits: int,
    *,
    num_steps: int,
    learning_rate: float,
    regularization: float,
    calibration_batch_size: int,
    max_affine_dim: int,
    generator: torch.Generator,
    calibration_inputs: Callable[[int], torch.Tensor] | None = None,
) -> AffineQuantizedLinear:
    """Build an :class:`AffineQuantizedLinear` from an ``nn.Linear``.

    Full-matrix affine optimization is used when ``out_features <= max_affine_dim``;
    otherwise the layer falls back to plain RTN (``A = I``).
    """
    weight = linear.weight.detach().to(torch.float32)
    bias = linear.bias.detach().to(torch.float32) if linear.bias is not None else None
    out_features, in_features = weight.shape
    if out_features <= max_affine_dim:
        if calibration_inputs is not None:
            calibration = calibration_inputs(in_features).to(torch.float32)
        else:
            calibration = torch.randn(calibration_batch_size, in_features, generator=generator, dtype=torch.float32)
        quantized_weight, a_inv_t = optimize_affine_transform(
            weight,
            calibration,
            bits,
            num_steps=num_steps,
            learning_rate=learning_rate,
            regularization=regularization,
        )
    else:
        quantized_weight = quantize_weight(weight, bits, straight_through=False)
        a_inv_t = torch.eye(out_features, dtype=torch.float32)
    out_dtype = linear.weight.dtype
    return AffineQuantizedLinear(
        quantized_weight.to(out_dtype),
        a_inv_t.to(out_dtype),
        None if bias is None else bias.to(out_dtype),
    )


def _replace_backbone_linears(parent: nn.Module, **kwargs) -> int:
    """Recursively replace leaf ``nn.Linear`` children with ``AffineQuantizedLinear``.

    Returns the number of linears replaced.
    """
    replaced = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, nn.Linear):
            setattr(parent, name, _make_affine_quantized_linear(child, **kwargs))
            replaced += 1
        elif len(list(child.children())) > 0:
            replaced += _replace_backbone_linears(child, **kwargs)
    return replaced


def apply_affine_quantization(
    model: nn.Module,
    *,
    bits: int = 4,
    num_calibration_steps: int = 25,
    calibration_batch_size: int = 8,
    learning_rate: float = 5e-3,
    regularization: float = 1e-3,
    max_affine_dim: int = 4096,
    seed: int = 0,
    calibration_inputs: Callable[[int], torch.Tensor] | None = None,
) -> int:
    """Apply affine post-training weight quantization to a model's backbone ``nn.Linear`` layers.

    Walks ``model.model.layers`` (the Llama transformer backbone) and replaces
    each ``nn.Linear`` projection with an :class:`AffineQuantizedLinear`. The
    flow-matching head, embeddings and ``lm_head`` are left untouched.

    This is an opt-in, in-place, one-time calibration pass — no retraining.

    Args:
        model: a model exposing a ``model.model.layers`` transformer backbone
            (e.g. :class:`tada.modules.tada.TadaForCausalLM`).
        bits: quantization bit-width.
        num_calibration_steps: Adam steps when fitting each layer's affine matrix.
        calibration_batch_size: number of synthetic calibration activations per layer
            (ignored when ``calibration_inputs`` is provided).
        learning_rate: Adam learning rate for the affine fit.
        regularization: L2 pull of ``A`` toward identity.
        max_affine_dim: layers wider than this fall back to plain RTN (full-matrix
            affine optimization is memory/compute-bound for very wide layers).
        seed: reproducibility seed for the synthetic calibration activations.
        calibration_inputs: optional callable mapping an ``in_features`` width to a
            ``[N, in_features]`` tensor of real captured activations for that layer.
            When provided, the affine is fit to real activations (faithful to the
            paper) instead of the synthetic Gaussian proxy.

    Returns:
        The number of backbone ``nn.Linear`` layers quantized.
    """
    backbone = getattr(getattr(model, "model", None), "layers", None)
    if backbone is None:
        raise AttributeError("apply_affine_quantization expects a model with a `.model.layers` transformer backbone")
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    kwargs = dict(
        bits=bits,
        num_steps=num_calibration_steps,
        learning_rate=learning_rate,
        regularization=regularization,
        calibration_batch_size=calibration_batch_size,
        max_affine_dim=max_affine_dim,
        generator=generator,
        calibration_inputs=calibration_inputs,
    )
    replaced = 0
    for layer in backbone:
        replaced += _replace_backbone_linears(layer, **kwargs)
    return replaced
