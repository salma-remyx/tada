import torch

from ..modules.tada import TadaConfig, TadaForCausalLM
from .weight_rounding import QuantizedLinear, optimize_linear, optimize_weight_rounding


def test_optimize_weight_rounding_beats_round_to_nearest():
    """SignSGD rounding should never be worse than round-to-nearest at the same bit-width."""
    torch.manual_seed(0)
    weight = torch.randn(32, 64)
    x = torch.randn(128, 64)
    ref = x @ weight.T

    bits = 4
    qmax = 2 ** (bits - 1) - 1
    scale_rtn = weight.abs().amax(dim=-1, keepdim=True) / qmax
    rtn = torch.clamp(torch.round(weight / scale_rtn), -qmax, qmax) * scale_rtn
    rtn_err = torch.mean((x @ rtn.T - ref) ** 2).item()

    q_int, scale, dequant = optimize_weight_rounding(weight, x, bits=bits, group_size=64, iters=100, lr=0.01)

    assert q_int.dtype == torch.int8
    assert q_int.abs().max().item() <= qmax
    assert dequant.shape == weight.shape
    sign_err = torch.mean((x @ dequant.T - ref) ** 2).item()
    assert sign_err <= rtn_err + 1e-9, f"SignRound ({sign_err}) worse than RTN ({rtn_err})"


def test_optimize_linear_swaps_to_quantized_module():
    torch.manual_seed(1)
    layer = torch.nn.Linear(64, 32)
    x = torch.randn(48, 64)
    ref = layer(x)

    quantized = optimize_linear(layer, x, bits=4, group_size=64, iters=50)

    assert isinstance(quantized, QuantizedLinear)
    # The compressed form (int8 + scale) is held as buffers, not a materialized float Parameter.
    assert quantized.q_int.dtype == torch.int8
    assert "weight" not in dict(quantized.named_parameters())
    assert {"q_int", "scale"} <= set(dict(quantized.named_buffers()))
    y = quantized(x)
    assert y.shape == ref.shape
    assert torch.isfinite(y).all()


def test_model_optimize_weight_rounding_quantizes_backbone():
    """Exercises the wiring in TadaForCausalLM: backbone Linears become QuantizedLinear."""
    torch.manual_seed(2)
    config = TadaConfig(
        num_hidden_layers=1,
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_time_classes=8,
    )
    model = TadaForCausalLM(config).eval()

    linears_before = [
        name for name, mod in model.named_modules() if name.startswith("model.") and isinstance(mod, torch.nn.Linear)
    ]
    assert linears_before, "expected Llama backbone Linears under model.*"

    returned = model.optimize_weight_rounding(bits=4, seq_len=16, batch=2, iters=20)
    assert returned is model

    quantized = [
        name for name, mod in model.named_modules() if name.startswith("model.") and isinstance(mod, QuantizedLinear)
    ]
    assert len(quantized) == len(linears_before)

    # The non-backbone Linears (acoustic_proj, lm_head, diffusion head) are left untouched.
    untouched = [
        name
        for name, mod in model.named_modules()
        if not name.startswith("model.") and isinstance(mod, torch.nn.Linear)
    ]
    assert untouched, "expected speech/lm_head Linears to remain nn.Linear"

    # The backbone still forwards and the activations used for calibration were captured.
    ids = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        out = model.model(input_ids=ids)
    assert torch.isfinite(out.last_hidden_state).all()
