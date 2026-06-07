from decimal import Decimal
from types import SimpleNamespace

from summarizer.llms import AnthropicSummarizer
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


def test_calculate_cost_uses_model_pricing():
    summarizer = _make_summarizer(client=None)
    assert summarizer.calculate_cost(1_000_000, 1_000_000) == Decimal("6")
