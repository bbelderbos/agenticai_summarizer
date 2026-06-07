import logging
from decimal import Decimal
from typing import Protocol

from anthropic import Anthropic
from decouple import config
from pydantic import ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    before_sleep_log,
)

from summarizer.exceptions import AnthropicAPIError
from summarizer.models import ScoredSummary, SummaryResponse
from summarizer.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

TOOL_NAME = "record_summary"
DEFAULT_MODEL = "claude-haiku-4-5"
MAX_ATTEMPTS = 3

PRICING_PER_MTOK: dict[str, tuple[Decimal, Decimal]] = {
    "claude-haiku-4-5": (Decimal("1"), Decimal("5")),
    "claude-sonnet-4-6": (Decimal("3"), Decimal("15")),
    "claude-opus-4-8": (Decimal("5"), Decimal("25")),
}


class Summarizer(Protocol):
    model: str

    def summarize(self, text: str) -> ScoredSummary: ...


class AnthropicSummarizer:
    def __init__(self, model: str | None = None, client: Anthropic | None = None):
        self.model = model or config("SUMMARIZER_MODEL", default=DEFAULT_MODEL)
        self.client = client or Anthropic(api_key=config("ANTHROPIC_API_KEY"))
        self._tool_schema = SummaryResponse.model_json_schema()

    @retry(
        stop=stop_after_attempt(MAX_ATTEMPTS),
        retry=retry_if_exception_type(ValidationError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def summarize(self, text: str) -> ScoredSummary:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[
                {
                    "name": TOOL_NAME,
                    "description": "Record the structured summary of the article.",
                    "input_schema": self._tool_schema,
                }
            ],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": text}],
        )

        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use is None:
            raise AnthropicAPIError("Anthropic returned no tool_use block")

        result = SummaryResponse.model_validate(tool_use.input)

        cost = self.calculate_cost(
            response.usage.input_tokens, response.usage.output_tokens
        )
        return ScoredSummary(response=result, cost=cost)

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> Decimal:
        try:
            input_rate, output_rate = PRICING_PER_MTOK[self.model]
        except KeyError:
            raise ValueError(
                f"no pricing configured for model {self.model!r}"
            ) from None
        return (
            Decimal(prompt_tokens) * input_rate
            + Decimal(completion_tokens) * output_rate
        ) / Decimal("1000000")
