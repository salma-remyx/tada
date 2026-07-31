"""Token confidence / calibration diagnostics for weight quantization.

Adapted (Mode 2) from:

    When Quantization Affects Confidence of Large Language Models?
    arXiv:2405.00632  (https://arxiv.org/abs/2405.00632)

That paper studies how post-training quantization / low-bit weight representation
shifts the confidence and calibration of LLM token predictions. This module ports
that *measurement* to the TADA backbone: given a model and a batch of text token
ids, it runs a reference forward pass, applies a low-bit round-to-nearest (RTN)
fake-quantization to every ``nn.Linear`` weight, runs a second forward pass, and
reports how token-level confidence, predictive entropy, top-1 agreement and (when
labels are supplied) expected calibration error moved between the two passes.

Ported from the paper at full fidelity (the core measurement):
    * confidence = max softmax probability per predicted token
    * calibration = Expected Calibration Error (ECE) over confidence bins
    * the pre-vs-post-quantization comparison itself

Substituted with target-native equivalents (auxiliaries, not core):
    * the paper's GPTQ / AWQ / per-scheme, multi-model sweep is replaced by a
      single dependency-free symmetric per-output-channel RTN fake-quantizer
      parameterized by ``bits`` (the canonical low-bit weight-PTQ baseline)
    * the paper's separate benchmark / eval harness is cut (evaluation belongs in
      a downstream PR); here we measure on whatever token batch the caller supplies
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import torch
from torch import nn

if TYPE_CHECKING:
    from ..modules.tada import TadaForCausalLM


@dataclass
class ConfidenceShiftReport:
    """Pre-vs-post-quantization shift in token confidence / calibration."""

    bits: int
    num_tokens: int
    mean_confidence_ref: float
    mean_confidence_quant: float
    confidence_delta: float
    mean_entropy_ref: float
    mean_entropy_quant: float
    entropy_delta: float
    top1_agreement: float
    ece_ref: float | None
    ece_quant: float | None
    ece_delta: float | None

    def __repr__(self) -> str:
        ece = "n/a"
        if self.ece_ref is not None and self.ece_quant is not None and self.ece_delta is not None:
            ece = f"{self.ece_ref:.4f} -> {self.ece_quant:.4f} (d={self.ece_delta:+.4f})"
        return (
            f"ConfidenceShiftReport(bits={self.bits}, tokens={self.num_tokens})\n"
            f"  mean confidence : {self.mean_confidence_ref:.4f} -> {self.mean_confidence_quant:.4f}"
            f" (d={self.confidence_delta:+.4f})\n"
            f"  mean entropy    : {self.mean_entropy_ref:.4f} -> {self.mean_entropy_quant:.4f}"
            f" (d={self.entropy_delta:+.4f})\n"
            f"  top-1 agreement : {self.top1_agreement:.4f}\n"
            f"  ECE             : {ece}"
        )


def softmax_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Max softmax probability per row, computed in float32 for stability."""
    probs = torch.softmax(logits.float(), dim=-1)
    return probs.amax(dim=-1)


def predictive_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Predictive entropy (nats) per row, computed in float32 for stability."""
    probs = torch.softmax(logits.float(), dim=-1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)


def expected_calibration_error(confidence: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error over ``n_bins`` equal-width confidence buckets.

    Args:
        confidence: per-token max softmax probability, shape (N,), values in [0, 1].
        correct: per-token correctness (1 if argmax == label, else 0), shape (N,).
        n_bins: number of equal-width bins over [0, 1].
    """
    confidence = confidence.float().flatten()
    correct = correct.float().flatten()
    if confidence.numel() == 0:
        return 0.0
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=confidence.device)
    ece = confidence.new_zeros(())
    for i in range(n_bins):
        lower, upper = edges[i], edges[i + 1]
        # Left-closed, right-open bins; fold the exact-0.0 mass into the first bin.
        in_bin = (confidence > lower) & (confidence <= upper)
        if i == 0:
            in_bin = in_bin | (confidence == 0.0)
        prop = in_bin.float().mean()
        if prop.item() > 0.0:
            acc = correct[in_bin].mean()
            conf = confidence[in_bin].mean()
            ece = ece + prop * (acc - conf).abs()
    return float(ece)


def fake_quantize_weight(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-output-channel round-to-nearest fake quantization.

    Quantizes each output row of ``weight`` (assumed (out, in)) to ``bits`` and
    returns the dequantized float tensor. Parameter-free, dependency-free.
    """
    if bits >= 16:
        return weight
    qmax = 2 ** (bits - 1) - 1
    w = weight.float()
    scale = w.abs().amax(dim=-1, keepdim=True) / qmax
    scale = scale.clamp_min(1e-8)
    return torch.round(w / scale) * scale


@contextmanager
def fake_quantized(module: nn.Module, bits: int) -> Iterator[None]:
    """Temporarily RTN-fake-quantize every ``nn.Linear`` weight in ``module``.

    Linear weights are mutated in place and restored exactly on exit, even when
    the wrapped block raises.
    """
    targets = [(linear, linear.weight.data.clone()) for linear in module.modules() if isinstance(linear, nn.Linear)]
    for linear, _ in targets:
        quantized = fake_quantize_weight(linear.weight.data, bits).to(linear.weight.dtype)
        linear.weight.data.copy_(quantized)
    try:
        yield
    finally:
        for linear, backup in targets:
            linear.weight.data.copy_(backup)


@torch.no_grad()
def collect_logits(
    model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Run the model's backbone over ``input_ids`` and return its prediction logits."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    return logits


@torch.no_grad()
def measure_confidence_shift(
    model: TadaForCausalLM,
    input_ids: torch.Tensor,
    *,
    bits: int = 8,
    labels: torch.Tensor | None = None,
    n_bins: int = 15,
    attention_mask: torch.Tensor | None = None,
) -> ConfidenceShiftReport:
    """Measure how low-bit weight quantization shifts token confidence/calibration.

    Runs a reference forward pass, RTN-fake-quantizes the backbone's Linear
    weights to ``bits`` bits, runs a second pass, and summarizes the shift.

    Args:
        model: a TadaForCausalLM (or any ``nn.Module`` exposing logits via forward).
        input_ids: teacher-forced text token ids, shape (batch, seq).
        bits: target weight bit-width for the quantized pass (e.g. 8, 4, 2).
        labels: optional next-token targets, shape (batch, seq). When supplied,
            ECE is computed for both passes; logits[:, :-1] predict labels[:, 1:].
        n_bins: number of ECE confidence bins.
        attention_mask: optional (batch, seq) mask; passed through to the model.

    Returns:
        A :class:`ConfidenceShiftReport` comparing the two passes.
    """
    ref_logits = collect_logits(model, input_ids, attention_mask=attention_mask)
    with fake_quantized(model, bits):
        quant_logits = collect_logits(model, input_ids, attention_mask=attention_mask)
    return _summarize_shift(ref_logits, quant_logits, labels=labels, n_bins=n_bins, bits=bits)


def _summarize_shift(
    ref_logits: torch.Tensor,
    quant_logits: torch.Tensor,
    *,
    labels: torch.Tensor | None,
    n_bins: int,
    bits: int,
) -> ConfidenceShiftReport:
    # Next-token prediction: position t predicts position t+1.
    ref_pred = ref_logits[:, :-1].reshape(-1, ref_logits.shape[-1]).float()
    quant_pred = quant_logits[:, :-1].reshape(-1, quant_logits.shape[-1]).float()

    conf_ref = softmax_confidence(ref_pred)
    conf_quant = softmax_confidence(quant_pred)
    ent_ref = predictive_entropy(ref_pred)
    ent_quant = predictive_entropy(quant_pred)
    agreement = (ref_pred.argmax(dim=-1) == quant_pred.argmax(dim=-1)).float().mean()

    ece_ref: float | None = None
    ece_quant: float | None = None
    ece_delta: float | None = None
    if labels is not None:
        target = labels[:, 1:].reshape(-1)
        correct_ref = (ref_pred.argmax(dim=-1) == target).float()
        correct_quant = (quant_pred.argmax(dim=-1) == target).float()
        ece_ref = expected_calibration_error(conf_ref, correct_ref, n_bins=n_bins)
        ece_quant = expected_calibration_error(conf_quant, correct_quant, n_bins=n_bins)
        ece_delta = ece_quant - ece_ref

    return ConfidenceShiftReport(
        bits=bits,
        num_tokens=int(ref_pred.shape[0]),
        mean_confidence_ref=float(conf_ref.mean()),
        mean_confidence_quant=float(conf_quant.mean()),
        confidence_delta=float(conf_quant.mean() - conf_ref.mean()),
        mean_entropy_ref=float(ent_ref.mean()),
        mean_entropy_quant=float(ent_quant.mean()),
        entropy_delta=float(ent_quant.mean() - ent_ref.mean()),
        top1_agreement=float(agreement),
        ece_ref=ece_ref,
        ece_quant=ece_quant,
        ece_delta=ece_delta,
    )
