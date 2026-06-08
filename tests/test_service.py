from decimal import Decimal
from unittest.mock import patch

import pytest

from summarizer.models import ScoredSummary, Sentiment, SummaryResponse
from summarizer.repo import InMemorySummaryRepo
from summarizer.service import SummarizerService


class FakeSummarizer:
    model = "fake-model"

    def summarize(self, text: str) -> ScoredSummary:
        return ScoredSummary(
            response=SummaryResponse(
                tl_dr="canned summary",
                key_points=["a", "b", "c"],
                tags=["t1", "t2", "t3"],
                reading_time_minutes=2,
                sentiment=Sentiment.NEUTRAL,
            ),
            cost=Decimal("0.01"),
        )


def _service(repo=None) -> SummarizerService:
    return SummarizerService(FakeSummarizer(), repo=repo)


def test_persist_false_stores_nothing():
    repo = InMemorySummaryRepo()
    with patch("summarizer.service.fetch_text", return_value="text"):
        result = _service(repo).summarize("hello", persist=False)
    assert result.persisted is False
    assert repo.get_all() == []


def test_persist_true_stores_one_row():
    repo = InMemorySummaryRepo()
    with patch("summarizer.service.fetch_text", return_value="some text"):
        _service(repo).summarize("hello", persist=True)
    rows = repo.get_all()
    assert len(rows) == 1
    assert rows[0].tl_dr == "canned summary"
    assert rows[0].tags == ["t1", "t2", "t3"]
    assert rows[0].model == "fake-model"


def test_persist_without_repo_raises():
    with patch("summarizer.service.fetch_text", return_value="text"):
        with pytest.raises(ValueError):
            _service().summarize("hello", persist=True)


def test_url_source_label_kept_text_label_otherwise():
    with patch("summarizer.service.fetch_text", return_value="text"):
        url_result = _service().summarize("https://example.com/post", persist=False)
        text_result = _service().summarize("just words", persist=False)
    assert url_result.source == "https://example.com/post"
    assert text_result.source == "text"
