"""Speech-reasoning multiple-choice evaluation for TADA.

Adapts the evaluation methodology of *SpeechR: A Benchmark for Speech
Reasoning in Large Audio-Language Models* (arXiv:2508.02018) to TADA's
audio path. SpeechR probes whether a large audio-language model can
*reason* over speech (temporal, spatial, speaker, and content reasoning)
rather than merely transcribe it. Each SpeechR item is an audio clip plus
a question and a fixed set of candidate answers, and the model is scored
by the candidate it assigns the lowest conditional loss to.

What is ported here is the *scoring methodology*, not the SpeechR dataset
or its per-task benchmark harness (those are downstream concerns): for
each candidate the scorer builds an audio + question + answer prompt,
runs ``TadaForCausalLM.generate``, extracts the mean cross-entropy over
the answer tokens using the same alignment convention as the existing
``run_*_tada.py`` evals, and reports per-category accuracy -- the shape
SpeechR uses to break reasoning ability down by dimension.

Attribution: SpeechR, arXiv:2508.02018v1.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from tada.modules.encoder import EncoderOutput
from tada.modules.tada import GenerationOutput, InferenceOptions

# TADA token framing accounts for BOS/EOS plus the acoustic shift; the
# existing run_*_tada.py evals trim the logits/labels accordingly before
# computing cross-entropy. See run_hellaswag_tada.py / run_sSC_tada.py.
_LOGIT_TRIM = 7
_LABEL_HEAD = 8
_LABEL_TAIL = 6

# Bucket name used by aggregate_by_category for the micro-average across
# every task category -- the headline number SpeechR reports on top of the
# per-dimension breakdown.
_OVERALL = "__overall__"


@dataclass
class SpeechRItem:
    """A single SpeechR multiple-choice item.

    Attributes:
        audio: input speech as a waveform tensor ``(channels, samples)`` or a
            path loadable by ``torchaudio``. The model reasons over this clip.
        question: the question the model must answer about the audio.
        candidates: ordered candidate answers; index ``label`` is the gold one.
        label: index into ``candidates`` of the correct answer.
        category: SpeechR task category (e.g. ``"temporal"``, ``"speaker"``)
            used to break down accuracy.
        sample_rate: sample rate of ``audio`` when it is a waveform.
    """

    audio: torch.Tensor | str
    question: str
    candidates: list[str]
    label: int
    category: str = "uncategorized"
    sample_rate: int = 24000


@dataclass
class ItemResult:
    """Outcome of scoring one :class:`SpeechRItem`."""

    category: str
    losses: list[float]
    prediction: int
    label: int

    @property
    def correct(self) -> bool:
        return self.prediction == self.label


@dataclass
class CategoryStats:
    """Per-category accuracy roll-up reported SpeechR-style."""

    category: str
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def text_conditioned_prompt(
    text: str,
    num_tokens: int,
    acoustic_dim: int,
    device: torch.device | str,
) -> EncoderOutput:
    """Build a text-conditioned EncoderOutput (zeroed acoustics).

    Mirrors the direct-construction path in ``run_hellaswag_tada.py``: no
    encoder or tokenizer round-trip is needed, so the caller supplies the
    precomputed ``num_tokens``. This is the path used for text-only scoring
    and is the one exercised by the unit tests.
    """
    return EncoderOutput(
        audio=torch.zeros(1, 0, device=device),
        audio_len=torch.zeros(1, device=device),
        text=[text],
        text_tokens_len=torch.tensor([num_tokens], device=device),
        token_positions=torch.zeros(1, num_tokens, dtype=torch.long, device=device),
        token_values=torch.zeros(1, num_tokens, acoustic_dim, device=device),
        token_masks=torch.zeros(1, num_tokens, dtype=torch.long, device=device),
    )


def answer_loss(outputs: GenerationOutput, context_len: int) -> float:
    """Mean cross-entropy over the candidate-answer token span.

    Follows the alignment convention of ``run_hellaswag_tada.py``: trim the
    TADA token framing, compute per-token cross-entropy, then average over
    the tokens *after* the ``context_len`` conditioning tokens (the audio +
    question), i.e. the answer span. This realizes the SpeechR scoring rule
    P(answer | audio, question): the lowest-loss candidate wins.
    """
    shift_logits = outputs.logits[..., _LOGIT_TRIM:-_LOGIT_TRIM, :].contiguous()
    shift_labels = outputs.input_text_ids[..., _LABEL_HEAD:-_LABEL_TAIL].contiguous()
    loss = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    # Degenerate short sequences can leave the answer span empty; fall back to
    # the whole-sequence mean rather than emit NaN.
    span = loss[context_len:] if context_len < loss.numel() else loss
    return span.mean().item()


def _build_prompt(
    item: SpeechRItem,
    candidate: str,
    context_len: int,
    acoustic_dim: int,
    device: torch.device | str,
    encoder,
    tokenize: Callable[[str], int],
) -> EncoderOutput:
    """Build the audio + question + answer prompt for one candidate.

    When ``encoder`` is given the speech is encoded through TADA's audio path
    (on-domain, like ``run_sSC_tada.py``); otherwise a text-only condition is
    used (zeroed acoustics, like ``run_hellaswag_tada.py``).
    """
    full_text = f"{item.question} {candidate}"
    num_tokens = tokenize(full_text)
    if encoder is None:
        return text_conditioned_prompt(full_text, num_tokens, acoustic_dim, device)
    audio = _as_waveform(item.audio, item.sample_rate).to(device)
    return encoder(
        audio=audio,
        text=[full_text],
        audio_length=torch.tensor([audio.shape[-1]], device=device),
        sample_rate=item.sample_rate,
        inference_window_size=30,
        inference_window_stride=28,
    )


def _as_waveform(audio: torch.Tensor | str, sample_rate: int) -> torch.Tensor:
    """Coerce ``audio`` to a batched waveform ``(1, channels, samples)``."""
    import torchaudio

    if isinstance(audio, torch.Tensor):
        wav = audio
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return wav.unsqueeze(0) if wav.dim() == 2 else wav
    wav, sr = torchaudio.load(str(audio))
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.unsqueeze(0)


def score_item(
    item: SpeechRItem,
    model,
    tokenize: Callable[[str], int],
    device: torch.device | str,
    *,
    encoder=None,
    inference_options: InferenceOptions | None = None,
) -> ItemResult:
    """Score every candidate of ``item``; return per-candidate losses + the argmin.

    The scorer builds an audio + question + answer prompt for each candidate
    and reads the conditional loss over the answer tokens from
    ``model.generate`` (SpeechR: lowest-loss candidate wins). ``tokenize`` maps
    a string to its token count and is injected so the scorer does not depend
    on any particular tokenizer at import time.
    """
    inference_options = inference_options or InferenceOptions(acoustic_cfg_scale=1.0)
    acoustic_dim = model.config.acoustic_dim
    context_len = tokenize(item.question)
    losses: list[float] = []
    for candidate in item.candidates:
        prompt = _build_prompt(item, candidate, context_len, acoustic_dim, device, encoder, tokenize)
        with torch.no_grad():
            outputs = model.generate(
                prompt,
                text="",
                num_transition_steps=0,
                num_extra_steps=0,
                inference_options=inference_options,
                use_text_in_prompt=True,
                normalize_text=False,
            )
        losses.append(answer_loss(outputs, context_len))
    prediction = losses.index(min(losses))
    return ItemResult(category=item.category, losses=losses, prediction=prediction, label=item.label)


def aggregate_by_category(results: list[ItemResult]) -> dict[str, CategoryStats]:
    """Roll item results into per-category accuracy plus an overall bucket.

    The ``"__overall__"`` key holds the micro-average across every category,
    matching the way SpeechR reports a headline number alongside the
    per-dimension breakdown.
    """
    stats: dict[str, CategoryStats] = {}
    overall = CategoryStats(_OVERALL)
    for result in results:
        bucket = stats.setdefault(result.category, CategoryStats(result.category))
        bucket.total += 1
        bucket.correct += int(result.correct)
        overall.total += 1
        overall.correct += int(result.correct)
    stats[_OVERALL] = overall
    return stats
