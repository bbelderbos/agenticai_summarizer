import logging
from decimal import Decimal
from typing import Protocol

from anthropic import Anthropic
from decouple import config
from pydantic import ValidationError

from summarizer.models import SummaryResponse
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

    def summarize(self, text: str) -> SummaryResponse: ...


class AnthropicSummarizer:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or config("SUMMARIZER_MODEL", default=DEFAULT_MODEL)
        self.client = Anthropic(api_key=api_key or config("ANTHROPIC_API_KEY"))
        self._tool_schema = SummaryResponse.model_json_schema()

    def summarize(self, text: str) -> SummaryResponse:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
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
                raise ValueError("Anthropic returned no tool_use block")

            try:
                result = SummaryResponse.model_validate(tool_use.input)
            except ValidationError:
                if attempt == MAX_ATTEMPTS:
                    logger.exception(
                        "tool input failed validation after %d attempts", MAX_ATTEMPTS
                    )
                    raise
                logger.warning(
                    "malformed tool input on attempt %d/%d, retrying",
                    attempt,
                    MAX_ATTEMPTS,
                )
                continue

            result.cost = self.calculate_cost(
                response.usage.input_tokens, response.usage.output_tokens
            )
            return result

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> Decimal:
        input_rate, output_rate = PRICING_PER_MTOK.get(
            self.model, (Decimal("1"), Decimal("5"))
        )
        return (
            Decimal(prompt_tokens) * input_rate
            + Decimal(completion_tokens) * output_rate
        ) / Decimal("1000000")
