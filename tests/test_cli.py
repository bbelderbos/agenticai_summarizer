from unittest.mock import MagicMock
import pytest
from summarizer.models import Summary
from summarizer.service import SummaryResult

# Target our imports from the actual package structure
from summarizer.cli import app


@pytest.fixture
def mock_dependencies(mocker):
    """Fixture to mock all external service/repo layers and configure Cyclopts for testing."""
    mock_repo_cls = mocker.patch("summarizer.cli.DBSummaryRepo")
    mock_service_cls = mocker.patch("summarizer.cli.SummarizerService")
    mock_config = mocker.patch("summarizer.cli.config")
    mock_console = mocker.patch("summarizer.cli.console")

    # Mock Anthropic to prevent Pydantic Validation errors from real invocations
    mocker.patch("summarizer.cli.AnthropicSummarizer")

    # FIX: Use a valid SQLAlchemy format string so create_engine doesn't crash
    mock_config.return_value = "sqlite:///:memory:"

    # Prevents Cyclopts from calling sys.exit() during test execution
    app.result_action = "return_value"

    return {
        "repo_cls": mock_repo_cls,
        "service_cls": mock_service_cls,
        "console": mock_console,
    }


def test_summarize_without_db(mock_dependencies, mocker):
    # Arrange
    mock_service = MagicMock()
    mock_dependencies["service_cls"].return_value = mock_service

    # Mock the internal structures returned by the service
    mock_response = MagicMock(
        tl_dr="This is a short summary.",
        key_points=["Point 1", "Point 2"],
        tags=["ai", "tech"],
        reading_time_minutes=2,
        sentiment="Positive",
    )
    mock_result = SummaryResult(
        source="http://example.com",
        response=mock_response,
        cost=0.0015,
        persisted=False,
    )
    mock_service.summarize.return_value = mock_result

    # Act
    app(["summarize", "http://example.com"])

    # Assert
    # Verify the database repo was not instantiated since db=False
    mock_dependencies["repo_cls"].assert_not_called()

    # Verify the service was spun up correctly without a repo dependency
    mock_dependencies["service_cls"].assert_called_once_with(mocker.ANY, repo=None)

    # Verify the business logic received the correct input arguments
    mock_service.summarize.assert_called_once_with("http://example.com", persist=False)

    # Verify that the rich table output was triggered
    mock_dependencies["console"].print.assert_called_once()


def test_summarize_with_db(mock_dependencies, mocker):
    # Arrange
    mock_repo = MagicMock()
    mock_dependencies["repo_cls"].return_value = mock_repo

    mock_service = MagicMock()
    mock_dependencies["service_cls"].return_value = mock_service

    mock_response = MagicMock(
        tl_dr="Saved summary.",
        key_points=["Point 1"],
        tags=["database"],
        reading_time_minutes=1,
        sentiment="Neutral",
    )
    mock_result = SummaryResult(
        source="http://example.com", response=mock_response, cost=0.0005, persisted=True
    )
    mock_service.summarize.return_value = mock_result

    # Act
    app(["summarize", "http://example.com", "--db"])

    # Assert
    # Repo should be initialized with the URL fetched from config()
    mock_dependencies["repo_cls"].assert_called_once_with("sqlite:///:memory:")

    # Service should be passed the active database repo instance
    mock_dependencies["service_cls"].assert_called_once_with(mocker.ANY, repo=mock_repo)
    mock_service.summarize.assert_called_once_with("http://example.com", persist=True)
    mock_dependencies["console"].print.assert_called_once()


def test_list_summaries_empty(mock_dependencies):
    # Arrange
    mock_repo = MagicMock()
    mock_repo.get_all.return_value = []
    mock_dependencies["repo_cls"].return_value = mock_repo

    # Act
    app(["list"])

    # Assert
    mock_repo.get_all.assert_called_once()
    mock_dependencies["console"].print.assert_called_once_with(
        "No saved summaries yet."
    )


def test_list_summaries_with_data(mock_dependencies):
    # Arrange
    mock_repo = MagicMock()

    # Mock an actual Summary model entry returned from database
    mock_summary = MagicMock(spec=Summary)
    mock_summary.id = 1
    mock_summary.source = "http://example.com"
    mock_summary.tl_dr = "A short story."
    mock_summary.sentiment = "Neutral"

    mock_repo.get_all.return_value = [mock_summary]
    mock_dependencies["repo_cls"].return_value = mock_repo

    # Act
    app(["list"])

    # Assert
    mock_repo.get_all.assert_called_once()
    # Confirms it generated a rich Table instance and called print
    mock_dependencies["console"].print.assert_called_once()
