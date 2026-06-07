from dataclasses import dataclass

from summarizer.fetch import fetch_text
from summarizer.llms import Summarizer
from summarizer.models import Summary, SummaryResponse
from summarizer.repo import SummaryRepository

EXCERPT_CHARS = 200


@dataclass(frozen=True)
class SummaryResult:
    response: SummaryResponse
    source: str
    persisted: bool


class SummarizerService:
    def __init__(self, summarizer: Summarizer, repo: SummaryRepository | None = None):
        self.summarizer = summarizer
        self.repo = repo

    def summarize(self, source: str, persist: bool = False) -> SummaryResult:
        text = fetch_text(source)
        response = self.summarizer.summarize(text)

        is_url = source.startswith(("http://", "https://"))
        label = source if is_url else "text"

        if persist and self.repo is not None:
            self.repo.add(
                Summary(
                    source=label,
                    source_excerpt=text[:EXCERPT_CHARS],
                    tl_dr=response.tl_dr,
                    key_points=response.key_points,
                    tags=response.tags,
                    reading_time_minutes=response.reading_time_minutes,
                    sentiment=response.sentiment,
                    model=self.summarizer.model,
                    cost=response.cost,
                )
            )

        return SummaryResult(response=response, source=label, persisted=persist)
