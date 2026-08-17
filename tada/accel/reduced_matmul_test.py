import pytest
import torch

from ..modules.tada import TadaConfig, TadaForCausalLM
from .reduced_matmul import ReducedMatmulLinear, reduced_matmul_stats


def _tiny_model() -> TadaForCausalLM:
    torch.manual_seed(0)
    config = TadaConfig(
        num_hidden_layers=2,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_time_classes=8,
        acoustic_dim=16,
    )
    return TadaForCausalLM(config).eval()


def _step(model: TadaForCausalLM, seed: int = 0):
    torch.manual_seed(seed)
    batch, seq = 2, 7
    input_ids = torch.randint(0, 64, (batch, seq))
    with torch.no_grad():
        return model.forward_one_step(
            input_ids,
            torch.zeros(batch, seq, 16),
            torch.zeros(batch, seq, dtype=torch.long),
            torch.zeros(batch, seq, dtype=torch.long),
            torch.zeros(batch, seq, dtype=torch.long),
            use_cache=False,
        )


def test_enable_reduced_matmul_wraps_backbone_and_records_flops():
    model = _tiny_model()
    replaced = model.enable_reduced_matmul(retention=0.75, components="attention")
    assert replaced, "expected attention projections to be wrapped"
    assert all(name.endswith("_proj") for name in replaced)
    assert all(isinstance(model.model.get_submodule(name), ReducedMatmulLinear) for name in replaced)

    _step(model)
    stats = model.reduced_matmul_stats()
    total = stats["total"]
    assert total["calls"] == len(replaced)  # one wrapped linear per layer per projection
    assert 0.0 < total["flops_fraction"] <= 1.0
    assert total["flops_fraction"] < 1.0  # reduction actually happened

    model.disable_reduced_matmul()
    assert reduced_matmul_stats(model.model)["total"]["calls"] == 0


def test_reduced_matmul_output_shape_and_full_retention_exact():
    model = _tiny_model()
    baseline = _step(model)

    model.enable_reduced_matmul(retention=0.5)
    reduced = _step(model)
    assert reduced.logits.shape == baseline.logits.shape
    assert not torch.allclose(reduced.logits, baseline.logits)

    model.disable_reduced_matmul()
    model.enable_reduced_matmul(retention=1.0)
    exact = _step(model)
    assert torch.allclose(exact.logits, baseline.logits, atol=1e-5)


def test_enable_reduced_matmul_validation():
    model = _tiny_model()
    with pytest.raises(ValueError):
        model.enable_reduced_matmul(retention=0.0)
    with pytest.raises(ValueError):
        model.enable_reduced_matmul(components="softmax")
    model.disable_reduced_matmul()  # no-op restore
