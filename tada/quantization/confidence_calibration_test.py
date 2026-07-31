import torch

from ..modules.tada import TadaConfig, TadaForCausalLM
from .confidence_calibration import expected_calibration_error, measure_confidence_shift


def _tiny_model(vocab_size: int = 320) -> TadaForCausalLM:
    # Tiny offline backbone (no weights download): mirrors the shape used by the
    # existing tada/modules/tada_test.py parametrize=None branch.
    config = TadaConfig(
        num_hidden_layers=1,
        vocab_size=vocab_size,
        hidden_size=16,
        num_attention_heads=1,
        num_time_classes=8,
    )
    return TadaForCausalLM(config).eval()


def test_measure_quantization_confidence_reports_shift_and_restores_weights():
    torch.manual_seed(0)
    model = _tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16))
    weight_before = next(model.parameters()).detach().clone()

    # Exercises the wiring: the public method on the existing TadaForCausalLM.
    report = model.measure_quantization_confidence(input_ids, bits=4, labels=input_ids)

    # RTN fake-quant is transient: backbone weights are restored exactly.
    assert torch.equal(weight_before, next(model.parameters()).detach())

    assert report.bits == 4
    assert report.num_tokens == 2 * 15  # batch * (seq - 1) next-token positions
    for value in (report.mean_confidence_ref, report.mean_confidence_quant):
        assert 0.0 <= value <= 1.0
    assert report.confidence_delta == report.mean_confidence_quant - report.mean_confidence_ref
    assert report.entropy_delta == report.mean_entropy_quant - report.mean_entropy_ref
    assert 0.0 <= report.top1_agreement <= 1.0
    assert report.ece_ref is not None and report.ece_quant is not None and report.ece_delta is not None
    for value in (report.ece_ref, report.ece_quant):
        assert 0.0 <= value <= 1.0


def test_aggressive_quantization_perturbs_predictions():
    # 2-bit RTN must move either confidence or the argmax on a random batch.
    torch.manual_seed(1)
    model = _tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (4, 12))
    report = measure_confidence_shift(model, input_ids, bits=2)
    perturbed = (abs(report.confidence_delta) > 0.0) or (report.top1_agreement < 1.0)
    assert perturbed


def test_measure_confidence_shift_without_labels_leaves_ece_none():
    model = _tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
    report = measure_confidence_shift(model, input_ids, bits=8)
    assert report.ece_ref is None and report.ece_quant is None and report.ece_delta is None


def test_expected_calibration_error_zero_when_perfectly_calibrated():
    confidence = torch.full((10,), 1.0)
    correct = torch.ones(10)
    assert expected_calibration_error(confidence, correct, n_bins=5) == 0.0


def test_expected_calibration_error_flags_overconfidence():
    # Confident everywhere but half wrong -> ECE near 0.5.
    confidence = torch.full((10,), 1.0)
    correct = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float32)
    ece = expected_calibration_error(confidence, correct, n_bins=5)
    assert 0.4 <= ece <= 0.6
