"""Activation-guided mixed-precision weight quantization.

Adapted (Mode 2) from Agile-Quant: Activation-Guided Quantization for Faster
Inference of LLMs on the Edge (Tu et al., 2023; arXiv:2312.05693).

Core mechanism (kept at full fidelity):
    Post-training weight quantization in which the bit-width of each
    ``nn.Linear`` is chosen from the *measured magnitude of its input
    activations* via a budget-constrained greedy allocation, instead of a single
    uniform bit-width across every layer. Layers that carry larger activation
    energy contribute more output error per quantized weight, so they are handed
    more bits under a fixed average-bit budget.

Substituted auxiliaries (Mode 2):
    * The paper's GPTQ weight-quantization engine is replaced by group-wise
      asymmetric round-to-nearest (RTN). GPTQ needs an external library plus a
      per-layer Hessian inverse; RTN is the parameter-free proxy that realizes
      the activation-guided bit-widths so the allocation can be measured.
    * The paper's bespoke sub-byte INT inference kernels are replaced by fake
      quantization: each weight tensor is quantized and dequantized back in
      place, so the model stays runnable in stock PyTorch. Packing weights into
      packed-int storage plus custom WxA16 matmul kernels is a deployment-engine
      concern and is intentionally out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn


@dataclass
class ActivationGuidedQuantReport:
    """Summary of an activation-guided quantization pass.

    ``avg_bits`` is the realized weighted-average bit-width across the targeted
    linears (``sum(bits_i * num_params_i) / total_params``); it is at most
    ``max_bits`` and tracks ``target_bits`` from below.
    """

    target_bits: float
    avg_bits: float
    compression_ratio: float
    num_linears: int
    min_bits: int
    max_bits: int
    group_size: int
    allocation: dict[str, int] = field(default_factory=dict)

    def bit_width_histogram(self) -> dict[int, int]:
        widths = list(self.allocation.values())
        return {bit: widths.count(bit) for bit in sorted(set(widths))}

    def __str__(self) -> str:
        return (
            f"ActivationGuidedQuant(target={self.target_bits:.2f}b, "
            f"avg={self.avg_bits:.3f}b, ~{self.compression_ratio:.2f}x vs fp16, "
            f"{self.num_linears} linears, histogram={self.bit_width_histogram()})"
        )


def _make_activation_hook(store: dict[str, float], name: str) -> Callable:
    def hook(_module: nn.Module, inputs: tuple, _output) -> None:
        x = inputs[0].detach()
        peak = float(x.abs().max())
        if peak > store.get(name, 0.0):
            store[name] = peak

    return hook


def _profile_activations(
    model: nn.Module,
    targets: list[tuple[str, nn.Module]],
    forward_fn: Callable,
    calibration_inputs,
) -> dict[str, float]:
    """Run one calibration forward and record each target's peak input magnitude."""
    store: dict[str, float] = {}
    handles = [mod.register_forward_hook(_make_activation_hook(store, name)) for name, mod in targets]
    try:
        with torch.no_grad():
            forward_fn(model, calibration_inputs)
    finally:
        for handle in handles:
            handle.remove()
    return store


def _allocate_bits(
    sensitivities: dict[str, float],
    sizes: dict[str, int],
    avg_bits: float,
    min_bits: int,
    max_bits: int,
) -> dict[str, int]:
    """Budget-constrained greedy bit allocation by marginal error reduction.

    Marginal reduction from raising layer ``i`` from ``b`` to ``b+1`` bits is
    proportional to ``sensitivity_i * 2**(-2b)`` (a convex-decreasing returns
    curve). Per unit of parameter cost, the layer with the largest value wins the
    next bit; this is the activation-guided counterpart of a uniform allocation.
    """
    names = list(sensitivities)
    bits = {name: min_bits for name in names}
    total = sum(sizes.values())
    budget = avg_bits * total
    used = sum(bits[name] * sizes[name] for name in names)
    while used < budget:
        best_name = None
        best_gain = 0.0
        for name in names:
            if bits[name] >= max_bits:
                continue
            if used + sizes[name] > budget:
                continue
            # 2**(-2b) == 4**(-b); the constant 0.75 factor is dropped (irrelevant to argmax).
            gain = sensitivities[name] * (4.0 ** (-bits[name])) / sizes[name]
            if gain > best_gain:
                best_gain = gain
                best_name = name
        if best_name is None:
            break
        bits[best_name] += 1
        used += sizes[best_name]
    return bits


def _quantize_weight_rtn(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Group-wise asymmetric round-to-nearest fake quantization of a Linear weight."""
    if bits >= 16:
        return weight
    max_level = (2**bits) - 1  # b bits -> integer codes 0..2**b-1 (2**b distinct levels)
    orig_dtype = weight.dtype
    w = weight.detach().float()
    in_features = w.shape[1]
    group = group_size if group_size > 0 else in_features
    result = torch.empty_like(w)
    for start in range(0, in_features, group):
        end = min(start + group, in_features)
        block = w[:, start:end]
        w_min = block.min()
        w_max = block.max()
        span = w_max - w_min
        if span.item() == 0.0:
            result[:, start:end] = block
            continue
        scale = span / max_level
        zero_point = torch.clamp(torch.round(-w_min / scale), 0, max_level)
        quantized = torch.clamp(torch.round(block / scale) + zero_point, 0, max_level)
        result[:, start:end] = (quantized - zero_point) * scale
    return result.to(orig_dtype)


def _default_forward(model: nn.Module, calibration_inputs) -> None:
    """Run a calibration step through a transformers-style causal LM backbone.

    Prefers ``model.model`` (the inner ``LlamaModel``), which executes the
    transformer blocks and trips the activation hooks without materializing the
    large LM-head projection; falls back to calling the module directly.
    """
    if hasattr(model, "model"):
        model.model(input_ids=calibration_inputs)
    else:
        model(calibration_inputs)


def activation_guided_quantize(
    model: nn.Module,
    calibration_inputs,
    avg_bits: float = 3.0,
    min_bits: int = 2,
    max_bits: int = 4,
    group_size: int = 128,
    forward_fn: Callable | None = None,
    module_prefix: str = "layers",
) -> ActivationGuidedQuantReport:
    """Quantize ``model``'s ``nn.Linear`` layers with activation-guided bit-widths.

    Targets every ``nn.Linear`` whose qualified name contains ``module_prefix``
    (default ``"layers"`` -> the transformer-block projections; embeddings and the
    LM head are left untouched). Returns a report describing the realized
    average bit-width and per-layer allocation.
    """
    if min_bits > max_bits:
        raise ValueError(f"min_bits ({min_bits}) must be <= max_bits ({max_bits})")
    if forward_fn is None:
        forward_fn = _default_forward

    targets = [
        (name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.Linear) and module_prefix in name
    ]
    if not targets:
        raise ValueError(
            f"No nn.Linear modules matching prefix {module_prefix!r} found on the model; "
            "pass module_prefix=<subtree> or check the model structure."
        )

    activations = _profile_activations(model, targets, forward_fn, calibration_inputs)

    sensitivities: dict[str, float] = {}
    sizes: dict[str, int] = {}
    for name, mod in targets:
        weight = mod.weight
        act_peak = activations.get(name, 0.0)
        weight_span = float((weight.max() - weight.min()).item())
        fan_in = weight.shape[1]
        # Output-error proxy: scales with activation energy, weight range^2, fan-in.
        sensitivities[name] = (act_peak**2) * (weight_span**2) * fan_in
        sizes[name] = weight.numel()

    allocation = _allocate_bits(sensitivities, sizes, avg_bits, min_bits, max_bits)

    with torch.no_grad():
        for name, mod in targets:
            mod.weight.copy_(_quantize_weight_rtn(mod.weight, allocation[name], group_size))

    total = sum(sizes.values())
    used = sum(bits * sizes[name] for name, bits in allocation.items())
    realized = used / total if total else 0.0
    return ActivationGuidedQuantReport(
        target_bits=float(avg_bits),
        avg_bits=realized,
        compression_ratio=(16.0 / realized) if realized > 0 else 0.0,
        num_linears=len(targets),
        min_bits=min_bits,
        max_bits=max_bits,
        group_size=group_size,
        allocation=allocation,
    )
