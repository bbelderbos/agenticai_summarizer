from typing import Annotated

from cyclopts import App, Parameter
from decouple import config
from rich.console import Console
from rich.table import Table

from summarizer.llms import AnthropicSummarizer
from summarizer.models import Summary
from summarizer.repo import DBSummaryRepo
from summarizer.service import SummarizerService, SummaryResult

app = App(name="summarizer", help="AI article/URL summarizer")
console = Console()


def _render_result(result: SummaryResult) -> None:
    response = result.response
    table = Table(title="Summary", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Source", result.source)
    table.add_row("TL;DR", response.tl_dr)
    table.add_row("Key points", "\n".join(f"• {p}" for p in response.key_points))
    table.add_row("Tags", ", ".join(response.tags))
    table.add_row("Reading time", f"{response.reading_time_minutes} min")
    table.add_row("Sentiment", response.sentiment)
    table.add_row("Cost", f"${result.cost:.6f}")
    table.add_row("Saved", "yes" if result.persisted else "no")
    console.print(table)


@app.command
def summarize(
    source: str,
    db: Annotated[bool, Parameter(help="Persist the summary to the database")] = False,
) -> None:
    """Summarize a URL or a chunk of text."""
    repo = DBSummaryRepo(config("DATABASE_URL")) if db else None
    service = SummarizerService(AnthropicSummarizer(), repo=repo)
    result = service.summarize(source, persist=db)
    _render_result(result)


@app.command(name="list")
def list_summaries() -> None:
    """List summaries saved in the database."""
    summaries: list[Summary] = DBSummaryRepo(config("DATABASE_URL")).get_all()
    if not summaries:
        console.print("No saved summaries yet.")
        return

    table = Table(title="Saved summaries")
    table.add_column("ID", style="cyan")
    table.add_column("Source", style="magenta", max_width=40)
    table.add_column("TL;DR", style="green", max_width=60)
    table.add_column("Sentiment")
    for s in summaries:
        table.add_row(str(s.id), s.source, s.tl_dr, s.sentiment)
    console.print(table)


if __name__ == "__main__":
    app()
