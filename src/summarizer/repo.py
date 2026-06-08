from abc import ABC, abstractmethod

from sqlmodel import Session, SQLModel, create_engine, select

from summarizer.models import Summary


class SummaryRepository(ABC):
    @abstractmethod
    def add(self, summary: Summary) -> Summary: ...

    @abstractmethod
    def get(self, summary_id: int) -> Summary | None: ...

    @abstractmethod
    def get_all(self) -> list[Summary]: ...

    @abstractmethod
    def delete(self, summary_id: int) -> None: ...


class InMemorySummaryRepo(SummaryRepository):
    def __init__(self) -> None:
        self._summaries: dict[int, Summary] = {}
        self._next_id = 1

    def add(self, summary: Summary) -> Summary:
        summary.id = self._next_id
        self._summaries[self._next_id] = summary
        self._next_id += 1
        return summary

    def get(self, summary_id: int) -> Summary | None:
        return self._summaries.get(summary_id)

    def get_all(self) -> list[Summary]:
        return list(self._summaries.values())

    def delete(self, summary_id: int) -> None:
        self._summaries.pop(summary_id, None)


class DBSummaryRepo(SummaryRepository):
    def __init__(self, db_url: str, session: Session | None = None):
        if session is None:
            engine = create_engine(db_url)
            SQLModel.metadata.create_all(engine)
            self.db = Session(engine)
        else:
            self.db = session

    def add(self, summary: Summary) -> Summary:
        self.db.add(summary)
        self.db.commit()
        self.db.refresh(summary)
        return summary

    def get(self, summary_id: int) -> Summary | None:
        return self.db.exec(select(Summary).where(Summary.id == summary_id)).first()

    def get_all(self) -> list[Summary]:
        return list(self.db.exec(select(Summary)).all())

    def delete(self, summary_id: int) -> None:
        summary = self.get(summary_id)
        if summary is not None:
            self.db.delete(summary)
            self.db.commit()
