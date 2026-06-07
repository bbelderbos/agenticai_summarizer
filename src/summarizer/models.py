from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field
from sqlalchemy import JSON, Column
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel, create_engine


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class SummaryResponse(BaseModel):
    tl_dr: str = Field(description="One or two sentence summary of the whole text")
    key_points: list[str] = Field(description="3-5 bullet-point takeaways")
    tags: list[str] = Field(description="3-6 lowercase topic tags")
    reading_time_minutes: int = Field(
        description="Estimated reading time of the source"
    )
    sentiment: Sentiment
    cost: Decimal = Field(
        default=Decimal("0"),
        description="Leave as 0 — set programmatically after the API call",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Summary(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    source: str
    source_excerpt: str
    tl_dr: str
    key_points: list[str] = SQLField(default_factory=list, sa_column=Column(JSON))
    tags: list[str] = SQLField(default_factory=list, sa_column=Column(JSON))
    reading_time_minutes: int
    sentiment: Sentiment
    model: str
    cost: Decimal = Decimal("0")
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))


def main() -> None:
    from decouple import config

    engine = create_engine(config("DATABASE_URL"))
    SQLModel.metadata.create_all(engine)
    print("database initialized")


if __name__ == "__main__":
    main()
