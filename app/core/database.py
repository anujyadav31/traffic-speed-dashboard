"""SQLAlchemy engine and session (SQLite, file set by DATABASE_URL)."""

import logging
from collections.abc import Iterator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    pass


_is_sqlite = settings.database_url.startswith("sqlite")
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False, "timeout": 30} if _is_sqlite else {},
    pool_pre_ping=True,
)

if _is_sqlite:
    # WAL lets the API read while the worker writes progress updates.
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    from app import models  # noqa: F401  (registers the tables)

    Base.metadata.create_all(engine)
    add_missing_columns()


def add_missing_columns() -> list[str]:
    """Add columns that exist in the models but not yet in the database file.

    create_all() only creates missing *tables*, so a database made by an older
    version of the app would otherwise lack new columns. Only nullable columns are
    added, so existing rows simply get NULL there. Returns the added "table.column"s.
    """
    added = []
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in existing or not col.nullable:
                    continue
                col_type = col.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {col_type}'))
                added.append(f"{table.name}.{col.name}")
    if added:
        logging.getLogger(__name__).info("Added database columns: %s", ", ".join(added))
    return added


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
