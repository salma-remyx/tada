"""Integration tests for affine weight quantization on the TADA Llama backbone.

These exercise the wiring (the ``TadaForCausalLM.apply_affine_quantization`` method
added in ``tada/modules/tada.py``) and assert the integrated behavior: backbone
``nn.Linear`` layers are swapped for ``AffineQuantizedLinear`` and the affine
transform reconstructs the original layer output at least as well as plain RTN.
"""

import pytest
import torch

from ..modules.tada import TadaConfig, TadaForCausalLM
from .affine_weight_quant import AffineQuantizedLinear, quantize_weight


def _tiny_config() -> TadaConfig:
    # Mirror the tiny-config construction used in tada_test.py; a small
    # intermediate_size keeps the MLP projections narrow (full affine path).
    return TadaConfig(
        num_hidden_layers=1,
        vocab_size=64,
        hidden_size=8,
        num_attention_heads=1,
        intermediate_size=16,
        num_time_classes=8,
    )


def test_apply_affine_quantization_replaces_backbone_linears():
    torch.manual_seed(0)
    model = TadaForCausalLM(_tiny_config()).eval()

    q_proj = model.model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, torch.nn.Linear)

    # Fixed calibration activations wide enough to cover every backbone in_features.
    # Feeding the SAME activations used to fit the affine makes the reconstruction
    # comparison well-defined (the affine is fit and evaluated on one set).
    layer_linears = [m for m in model.model.layers[0].modules() if isinstance(m, torch.nn.Linear)]
    max_in_features = max(m.in_features for m in layer_linears)
    calibration = torch.randn(32, max_in_features)

    q_calib = calibration[:, : q_proj.in_features]
    with torch.no_grad():
        reference_output = q_proj(q_calib)

    # Plain RTN baseline (A = I, no affine transform): the quantity AffineQuant beats.
    rtn_weight = quantize_weight(q_proj.weight.detach().float(), bits=4, straight_through=False)
    with torch.no_grad():
        rtn_output = q_calib @ rtn_weight.t() + (q_proj.bias if q_proj.bias is not None else 0.0)
    rtn_error = (rtn_output - reference_output).pow(2).mean().item()

    # Exercise the wiring: the public method delegates into the capability module,
    # fitting each layer's affine transform to the supplied calibration activations.
    returned = model.apply_affine_quantization(
        bits=4, seed=0, calibration_inputs=lambda in_features: calibration[:, :in_features]
    )
    assert returned is model  # opt-in, in-place, chainable

    quantized = model.model.layers[0].self_attn.q_proj
    assert isinstance(quantized, AffineQuantizedLinear)
    # Every backbone nn.Linear projection should have been swapped out.
    assert sum(1 for m in model.model.layers[0].modules() if isinstance(m, torch.nn.Linear)) == 0

    # Forward contract under quantization, and the affine result vs RTN.
    with torch.no_grad():
        affine_output = quantized(q_calib)
    affine_error = (affine_output - reference_output).pow(2).mean().item()

    # The optimized affine transform is best-checkpointed from the RTN (A = I)
    # starting point on this calibration set, so it must be no worse than plain RTN.
    assert affine_error <= rtn_error + 1e-12

    # Sanity: output magnitude is preserved (compensation didn't blow up).
    assert torch.isfinite(affine_output).all()
    assert affine_output.std().item() > 0


def test_apply_affine_quantization_is_reproducible():
    torch.manual_seed(123)
    model_a = TadaForCausalLM(_tiny_config()).eval()
    torch.manual_seed(123)
    model_b = TadaForCausalLM(_tiny_config()).eval()

    model_a.apply_affine_quantization(bits=4, seed=7)
    model_b.apply_affine_quantization(bits=4, seed=7)

    wa = model_a.model.layers[0].mlp.down_proj.weight
    wb = model_b.model.layers[0].mlp.down_proj.weight
    assert torch.allclose(wa, wb)


def test_apply_affine_quantization_preserves_full_forward():
    torch.manual_seed(42)
    model = TadaForCausalLM(_tiny_config()).eval()

    input_ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        reference = model.model(input_ids).last_hidden_state.clone()

    model.apply_affine_quantization(bits=4, seed=0)

    with torch.no_grad():
        quantized = model.model(input_ids).last_hidden_state

    # The flow-matching head is untouched; the quantized backbone still produces a
    # finite hidden state of the same shape.
    assert quantized.shape == reference.shape
    assert torch.isfinite(quantized).all()
    relative_error = (quantized - reference).norm() / reference.norm()
    assert relative_error < 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
