"""A database made by an older version of the app is upgraded in place, keeping its rows."""

import sqlite3

from sqlalchemy import create_engine, inspect

from app.core import database
from tests.conftest import TMP_DIR


def test_old_database_gets_new_columns(monkeypatch):
    path = TMP_DIR / "old_version.db"
    con = sqlite3.connect(path)
    # The samples/jobs tables as the first version of the app created them.
    con.executescript(
        """
        CREATE TABLE jobs (id VARCHAR(32) PRIMARY KEY, name VARCHAR(200), source VARCHAR(20),
            status VARCHAR(20), config JSON, progress FLOAT, frames_done INTEGER,
            frames_total INTEGER, live_stats JSON, has_video BOOLEAN, cancel_requested BOOLEAN,
            created_at DATETIME);
        CREATE TABLE samples (id INTEGER PRIMARY KEY, job_id VARCHAR(32), vehicle_id VARCHAR(32),
            frame_idx INTEGER, t FLOAT, s FLOAT, l FLOAT, speed FLOAT);
        INSERT INTO samples (job_id, vehicle_id, frame_idx, t, s, l, speed)
            VALUES ('abc', '7', 0, 0.0, 12.5, -1.0, 20.0);
        """
    )
    con.commit()
    con.close()

    engine = create_engine(f"sqlite:///{path}")
    monkeypatch.setattr(database, "engine", engine)
    added = database.add_missing_columns()

    assert "samples.x_px" in added and "samples.confidence" in added
    assert "jobs.location" in added and "jobs.run_info" in added
    cols = {c["name"] for c in inspect(engine).get_columns("samples")}
    assert {"x_px", "y_px", "bbox_x1", "bbox_y2", "confidence", "class_raw"} <= cols
    with engine.connect() as conn:
        row = conn.exec_driver_sql("SELECT s, speed, x_px FROM samples").one()
    assert tuple(row) == (12.5, 20.0, None)  # old data kept, new columns empty
    assert database.add_missing_columns() == []  # running again changes nothing
