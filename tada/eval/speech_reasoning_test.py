"""Unit tests for the SpeechR speech-reasoning scorer.

The selection / aggregation logic is exercised against TADA's real data
structures (``EncoderOutput``, ``GenerationOutput``, ``InferenceOptions``
imported from the existing ``tada.modules.*``) using a deterministic stub
model in place of the multi-GB ``TadaForCausalLM`` weights, which cannot be
downloaded in a unit test. The stub returns real ``GenerationOutput``
objects, so the loss extraction and answer-span slicing run against the
production code path.
"""

import torch

from tada.eval.speech_reasoning import (
    CategoryStats,
    ItemResult,
    SpeechRItem,
    aggregate_by_category,
    answer_loss,
    score_item,
    text_conditioned_prompt,
)
from tada.modules.encoder import EncoderOutput
from tada.modules.tada import GenerationOutput, InferenceOptions

# Fixed geometry for the stub outputs. seq_len must exceed 2 * _LOGIT_TRIM so
# the alignment slices in answer_loss leave a non-empty answer span.
SEQ_LEN = 24
VOCAB = 64
ACOUSTIC_DIM = 8
_DEVICE = torch.device("cpu")


def _aligned_labels() -> torch.Tensor:
    """Deterministic label sequence for the stub GenerationOutput."""
    return (torch.arange(SEQ_LEN) % (VOCAB - 1) + 1).view(1, SEQ_LEN)


def _peaky_logits() -> torch.Tensor:
    """Logits whose t-th row peaks on the (t+1)-th label: causal next-token
    alignment, so the trimmed slices in answer_loss agree -> near-zero CE."""
    target = torch.roll(_aligned_labels()[0], shifts=-1)
    return torch.nn.functional.one_hot(target, VOCAB).float() * 50.0


class _LossOrderedStubModel:
    """Stand-in for TadaForCausalLM whose generate() loss is text-driven.

    If the prompt text marks the candidate as correct (contains "CORRECT"),
    the returned logits peak on the next token so cross-entropy is ~0;
    otherwise logits are uniform so cross-entropy is ~log(vocab). This lets
    the scorer's argmin selection be asserted deterministically.
    """

    class _Config:
        acoustic_dim = ACOUSTIC_DIM

    def __init__(self):
        self.config = self._Config()

    def generate(self, prompt: EncoderOutput, **kwargs) -> GenerationOutput:
        del kwargs  # the real generate takes transition/step/options kwargs we ignore here

        labels = _aligned_labels()
        logits = _peaky_logits() if "CORRECT" in prompt.text[0] else torch.zeros(SEQ_LEN, VOCAB)
        return GenerationOutput(logits=logits.view(1, SEQ_LEN, VOCAB), input_text_ids=labels)


def _tokenize(text: str) -> int:
    """Deterministic token counter that needs no HF tokenizer."""
    return max(1, len(text.split()))


def test_text_conditioned_prompt_builds_real_encoder_output():
    prompt = text_conditioned_prompt("question answer", num_tokens=4, acoustic_dim=ACOUSTIC_DIM, device=_DEVICE)

    assert isinstance(prompt, EncoderOutput)
    assert prompt.text == ["question answer"]
    assert prompt.token_positions.shape == (1, 4)
    # Acoustic channel width matches the model's acoustic_dim, as in run_hellaswag_tada.py.
    assert prompt.token_values.shape == (1, 4, ACOUSTIC_DIM)
    assert prompt.text_tokens_len.item() == 4


def test_answer_loss_orders_correct_below_wrong():
    labels = _aligned_labels()
    correct = GenerationOutput(logits=_peaky_logits().view(1, SEQ_LEN, VOCAB), input_text_ids=labels)
    wrong = GenerationOutput(logits=torch.zeros(1, SEQ_LEN, VOCAB), input_text_ids=labels)

    correct_loss = answer_loss(correct, context_len=2)
    wrong_loss = answer_loss(wrong, context_len=2)

    assert correct_loss < 0.5, correct_loss
    assert wrong_loss > 1.0, wrong_loss
    assert correct_loss < wrong_loss


def test_answer_loss_falls_back_when_answer_span_empty():
    # context_len past the end of the trimmed span must not raise or return NaN.
    outputs = GenerationOutput(logits=torch.zeros(1, SEQ_LEN, VOCAB), input_text_ids=_aligned_labels())
    loss = answer_loss(outputs, context_len=10_000)
    assert not torch.isnan(torch.tensor(loss))


def test_score_item_picks_lowest_loss_candidate():
    model = _LossOrderedStubModel()
    options = InferenceOptions(acoustic_cfg_scale=1.0)

    item_a = SpeechRItem("ignored.wav", "q", ["CORRECT a", "WRONG b"], 0, "temporal")
    item_b = SpeechRItem("ignored.wav", "q", ["WRONG c", "CORRECT d"], 1, "speaker")

    result_a = score_item(item_a, model, _tokenize, _DEVICE, inference_options=options)
    result_b = score_item(item_b, model, _tokenize, _DEVICE, inference_options=options)

    assert len(result_a.losses) == 2
    assert result_a.prediction == 0 and result_a.correct
    assert result_b.prediction == 1 and result_b.correct
    # The correct candidate is always scored lowest.
    assert result_a.losses.index(min(result_a.losses)) == result_a.label
    assert result_b.losses.index(min(result_b.losses)) == result_b.label


def test_aggregate_by_category_reports_per_category_and_overall():
    results = [
        ItemResult(category="temporal", losses=[0.1, 0.9], prediction=0, label=0),  # correct
        ItemResult(category="temporal", losses=[0.9, 0.1], prediction=0, label=1),  # wrong
        ItemResult(category="speaker", losses=[0.1, 0.9], prediction=0, label=0),  # correct
    ]
    stats = aggregate_by_category(results)

    assert isinstance(stats["temporal"], CategoryStats)
    assert stats["temporal"].correct == 1 and stats["temporal"].total == 2
    assert stats["speaker"].correct == 1 and stats["speaker"].total == 1
    assert stats["__overall__"].correct == 2 and stats["__overall__"].total == 3
    assert stats["__overall__"].accuracy == 2 / 3
