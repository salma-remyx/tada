"""Dense-Jump time schedule for flow-matching ODE inference.

Adapted from "Dense-Jump Flow Matching with Non-Uniform Time Scheduling for
Robotic Policies: Mitigating Multi-Step Inference Degradation", Zidong Chen
et al., arXiv:2509.13574. Reference implementation (MIT):
https://github.com/ZidongChen25/Dense-Jump-FlowMatchingPolicy
"""

import torch

# Default endpoint of the dense region. Steps are spent uniformly in
# [0, DEFAULT_JUMP_POINT]; the final step then jumps to t=1.
DEFAULT_JUMP_POINT = 0.75


def build_dense_jump_schedule(
    num_steps: int,
    jump_point: float | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a Dense-Jump time schedule for flow-matching ODE integration.

    The paper observes that, for rectified-flow / flow-matching policies,
    increasing the number of uniformly spaced Euler inference steps
    counter-intuitively *degrades* the output. The cause it identifies is that
    uniformly spaced steps oversample the late-time region (t -> 1) where the
    learned velocity field becomes non-Lipschitz and unstable. The Dense-Jump
    schedule avoids that region: it spends ``(num_steps - 1)`` uniformly
    spaced steps across the well-behaved ``[0, jump_point]`` interval and
    replaces the unstable approach to ``t = 1`` with a single large jump
    step. The total number of Euler steps is unchanged (``num_steps``), but
    none are wasted in the unstable tail, so fewer steps can be used without
    the quality loss that uniform over-stepping introduces.

    This maps directly onto TADA's acoustic flow-matching head, whose stated
    optimization direction is "reduce flow matching steps / improve the
    real-time factor": dense-jump lets a small step budget concentrate where
    the velocity field is well-behaved.

    Args:
        num_steps: Total number of Euler steps (>= 2). ``(num_steps - 1)``
            steps cover ``[0, jump_point]`` and one final step jumps to 1.
        jump_point: Endpoint of the dense region, strictly in ``(0, 1)``.
            ``None`` falls back to :data:`DEFAULT_JUMP_POINT`.
        device: Torch device for the returned tensor.

    Returns:
        Tensor of shape ``(num_steps + 1,)`` with values in ``[0, 1]`` and
        exact endpoints ``0.0`` and ``1.0``, matching the contract of
        :meth:`tada.modules.tada.TadaForCausalLM._build_time_schedule`.
    """
    if num_steps < 2:
        raise ValueError(f"dense_jump schedule requires num_steps >= 2, got {num_steps}")
    jp = DEFAULT_JUMP_POINT if jump_point is None else jump_point
    if not (0.0 < jp < 1.0):
        raise ValueError(f"dense_jump jump_point must be strictly in (0, 1), got {jp}")

    # (num_steps - 1) uniform Euler steps across [0, jp]: linspace emits
    # num_steps points (including both endpoints) -> num_steps - 1 intervals.
    left = torch.linspace(0.0, jp, num_steps, device=device)
    # Append the single jump anchor at t=1; the final step (jp -> 1) skips the
    # non-Lipschitz region near t=1 that uniform over-sampling falls into.
    t_span = torch.cat([left, torch.tensor([1.0], device=device)])
    t_span[0] = 0.0
    t_span[-1] = 1.0
    return t_span
