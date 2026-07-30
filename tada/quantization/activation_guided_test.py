import torch
from torch import nn

from ..modules.tada import TadaConfig, TadaForCausalLM
from .activation_guided import _quantize_weight_rtn, activation_guided_quantize


class _TwoLinears(nn.Module):
    """Two independent linears so a calibration forward can feed each a different input."""

    def __init__(self):
        super().__init__()
        self.layers_loud = nn.Linear(8, 8)
        self.layers_quiet = nn.Linear(8, 8)

    def forward(self, x_loud, x_quiet):
        self.layers_loud(x_loud)
        self.layers_quiet(x_quiet)


def test_quantize_weight_rtn_collapses_to_few_levels():
    torch.manual_seed(0)
    bits, group_size = 2, 8
    weight = torch.randn(4, 16)
    quantized = _quantize_weight_rtn(weight, bits=bits, group_size=group_size)

    assert quantized.shape == weight.shape
    num_levels = 2**bits
    for group in range(weight.shape[1] // group_size):
        block_w = weight[:, group * group_size : (group + 1) * group_size]
        block_q = quantized[:, group * group_size : (group + 1) * group_size]
        # Each group dequantizes to at most 2**bits distinct grid points.
        assert len(torch.unique(block_q)) <= num_levels
        # Reconstruction error is bounded by one quantization step per group.
        step = (block_w.max() - block_w.min()) / (num_levels - 1)
        assert torch.abs(block_q - block_w).max() <= step + 1e-5


def test_activation_guided_allocates_more_bits_to_larger_activations():
    """Core Agile-Quant insight: a layer with larger activation energy gets more bits."""
    torch.manual_seed(123)
    model = _TwoLinears()
    # Equalize the weights so the two layers differ ONLY in their activation magnitude.
    model.layers_quiet.weight.data.copy_(model.layers_loud.weight.data)
    model.layers_quiet.bias.data.copy_(model.layers_loud.bias.data)
    loud = torch.randn(4, 8) * 1e3
    quiet = torch.randn(4, 8) * 1e-3

    def _run(_model, _inputs):
        _model(loud, quiet)

    report = activation_guided_quantize(
        model, None, avg_bits=3.0, min_bits=2, max_bits=4, group_size=8, forward_fn=_run
    )

    assert report.allocation["layers_loud"] > report.allocation["layers_quiet"]
    assert report.min_bits <= min(report.allocation.values())
    assert max(report.allocation.values()) <= report.max_bits
    assert abs(report.avg_bits - 3.0) < 1e-6
    assert report.compression_ratio > 0.0


def test_tada_apply_activation_guided_quantization():
    """Integration: the TadaForCausalLM call site wires the pass onto the Llama backbone."""
    torch.manual_seed(0)
    config = TadaConfig(
        num_hidden_layers=2,
        vocab_size=128256,
        hidden_size=8,
        num_attention_heads=1,
        intermediate_size=32,
        num_time_classes=8,
    )
    model = TadaForCausalLM(config).eval()
    input_ids = torch.randint(0, 1280, (1, 16))

    q_proj = next(m for n, m in model.named_modules() if n == "model.layers.0.self_attn.q_proj")
    weight_before = q_proj.weight.detach().clone()

    report = model.apply_activation_guided_quantization(input_ids, avg_bits=3.0, min_bits=2, max_bits=4, group_size=8)

    # 2 transformer layers x (4 attention + 3 MLP) projections.
    assert report.num_linears == 2 * (4 + 3)
    assert report.min_bits <= min(report.allocation.values())
    assert max(report.allocation.values()) <= report.max_bits
    assert 0 < report.avg_bits <= report.max_bits
    # Only transformer-block linears are targeted; embeddings / lm_head are left alone.
    assert all("layers" in name for name in report.allocation)
    assert "lm_head" not in report.allocation
    # Fake quantization changed the targeted weight in place.
    assert not torch.allclose(weight_before, q_proj.weight.detach())
    # The backbone still runs end to end after the in-place quantization.
    with torch.no_grad():
        model.model(input_ids=input_ids)
