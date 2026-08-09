import pytest
import torch

from ..modules.tada import InferenceOptions, TadaForCausalLM
from .dense_jump_schedule import DEFAULT_JUMP_POINT, build_dense_jump_schedule


def test_build_dense_jump_schedule_structure():
    # 4 steps -> 3 uniform steps in [0, 0.6] then one jump of 0.4 to t=1.
    t_span = build_dense_jump_schedule(num_steps=4, jump_point=0.6)

    assert t_span.shape == (5,)
    assert t_span[0].item() == 0.0
    assert t_span[-1].item() == 1.0
    # Monotonically non-decreasing.
    assert torch.all(t_span[1:] >= t_span[:-1])

    diffs = t_span[1:] - t_span[:-1]
    dense = diffs[:-1]  # first num_steps - 1 intervals
    assert torch.allclose(dense, dense[0].expand_as(dense))  # uniform in [0, jp]
    assert torch.allclose(dense[0], torch.tensor(0.6 / 3))
    # The final jump step is the (1 - jp) leap to t=1, larger than a dense step.
    assert torch.allclose(diffs[-1], torch.tensor(0.4))
    assert diffs[-1].item() > dense[0].item()


def test_build_dense_jump_schedule_default_jump_point():
    t_span = build_dense_jump_schedule(num_steps=10)
    assert t_span.shape == (11,)
    assert torch.allclose(t_span[-2], torch.tensor(DEFAULT_JUMP_POINT))


@pytest.mark.parametrize("num_steps", [1, 0])
def test_build_dense_jump_schedule_rejects_too_few_steps(num_steps: int):
    with pytest.raises(ValueError, match="num_steps"):
        build_dense_jump_schedule(num_steps=num_steps)


@pytest.mark.parametrize("jp", [0.0, 1.0, 1.5, -0.1])
def test_build_dense_jump_schedule_rejects_bad_jump_point(jp: float):
    with pytest.raises(ValueError, match="jump_point"):
        build_dense_jump_schedule(num_steps=4, jump_point=jp)


def test_wired_into_build_time_schedule():
    """The 'dense_jump' option flows through the (non-new) model class's
    schedule builder exactly like the existing uniform/cosine/logsnr options."""
    t_span = TadaForCausalLM._build_time_schedule(
        num_steps=4, schedule="dense_jump", device=torch.device("cpu"), jump_point=0.6
    )
    assert t_span.shape == (5,)
    assert torch.allclose(t_span, build_dense_jump_schedule(4, 0.6, torch.device("cpu")))


def test_wired_through_inference_options():
    """InferenceOptions carries the new option into the schedule builder."""
    opts = InferenceOptions(time_schedule="dense_jump", time_schedule_jump_point=0.6)
    assert opts.time_schedule == "dense_jump"

    t_span = TadaForCausalLM._build_time_schedule(
        num_steps=opts.num_flow_matching_steps,
        schedule=opts.time_schedule,
        device=torch.device("cpu"),
        jump_point=opts.time_schedule_jump_point,
    )
    assert t_span.shape == (opts.num_flow_matching_steps + 1,)
    # Last dense point sits exactly at the jump point, then a single leap to 1.
    assert torch.allclose(t_span[-2], torch.tensor(0.6))
    assert torch.allclose(t_span[-1], torch.tensor(1.0))
