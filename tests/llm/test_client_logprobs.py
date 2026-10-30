"""Log-probabilities from the LLM client: the recorder and the span helper (no model needed)."""

import math

import pytest

from trialguide.llm.client import span_logprob


def test_span_logprob_sums_the_tokens_overlapping_the_span():
    tokens = [['{"', -1.0], ["status", -0.1], ['": "', -0.2], ["not", -0.3], ["_met", -0.4], ['"}', -0.5]]
    text = "".join(t for t, _ in tokens)
    start = text.index("not_met")
    assert span_logprob(tokens, start, start + len("not_met")) == pytest.approx(-0.7)
    assert math.exp(span_logprob(tokens, 0, 0)) == 1.0          # empty span: probability 1


def test_recorder_keeps_the_greedy_tokens_logprob_and_leaves_scores_unchanged():
    torch = pytest.importorskip("torch")
    from trialguide.llm.client import _GreedyLogprobs
    rec = _GreedyLogprobs()
    scores = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    assert torch.equal(rec(None, scores), scores)
    expected = torch.log_softmax(scores, dim=-1).max(dim=-1).values
    assert torch.allclose(rec.steps[0], expected)
