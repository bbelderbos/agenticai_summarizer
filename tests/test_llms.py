from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from summarizer.llms import MAX_ATTEMPTS, AnthropicSummarizer
from summarizer.models import Sentiment


def _tool_input() -> dict:
    return {
        "tl_dr": "A short summary.",
        "key_points": ["point one", "point two"],
        "tags": ["python", "ai"],
        "reading_time_minutes": 4,
        "sentiment": "positive",
    }


class FakeAnthropicClient:
    def __init__(self, tool_input: dict, input_tokens: int, output_tokens: int):
        self._response = SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", input=tool_input)],
            usage=SimpleNamespace(
                input_tokens=input_tokens, output_tokens=output_tokens
            ),
        )
        self.messages = SimpleNamespace(create=lambda **_: self._response)


class SequenceAnthropicClient:
    def __init__(
        self,
        tool_inputs: list[dict],
        input_tokens: int = 1000,
        output_tokens: int = 200,
    ):
        self._responses = [
            SimpleNamespace(
                content=[SimpleNamespace(type="tool_use", input=tool_input)],
                usage=SimpleNamespace(
                    input_tokens=input_tokens, output_tokens=output_tokens
                ),
            )
            for tool_input in tool_inputs
        ]
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **_):
        response = self._responses[self.calls]
        self.calls += 1
        return response


def _make_summarizer(client) -> AnthropicSummarizer:
    summarizer = AnthropicSummarizer.__new__(AnthropicSummarizer)
    summarizer.model = "claude-haiku-4-5"
    summarizer.client = client
    summarizer._tool_schema = {}
    return summarizer


def test_summarize_parses_tool_use_into_model():
    client = FakeAnthropicClient(_tool_input(), input_tokens=1000, output_tokens=200)
    result = _make_summarizer(client).summarize("some article text")

    assert result.tl_dr == "A short summary."
    assert result.key_points == ["point one", "point two"]
    assert result.sentiment is Sentiment.POSITIVE
    # 1000 in @ $1/Mtok + 200 out @ $5/Mtok = 0.001 + 0.001 = 0.002
    assert result.cost == Decimal("0.002")


def test_summarize_retries_on_malformed_tool_input():
    malformed = {
        "tl_dr": "x",
        "key_points": '<parameter name="key_points"><item>oops</item>',
    }
    client = SequenceAnthropicClient([malformed, _tool_input()])

    result = _make_summarizer(client).summarize("some article text")

    assert client.calls == 2
    assert result.key_points == ["point one", "point two"]


def test_summarize_raises_after_exhausting_retries():
    malformed = {"tl_dr": "x", "key_points": "<item>oops</item>"}
    client = SequenceAnthropicClient([malformed] * MAX_ATTEMPTS)

    with pytest.raises(ValidationError):
        _make_summarizer(client).summarize("some article text")

    assert client.calls == MAX_ATTEMPTS


def test_calculate_cost_uses_model_pricing():
    summarizer = _make_summarizer(client=None)
    assert summarizer.calculate_cost(1_000_000, 1_000_000) == Decimal("6")
