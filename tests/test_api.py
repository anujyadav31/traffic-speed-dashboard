import io
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.pipeline.analysis import analyse
from app.pipeline.config import PipelineConfig
from app.pipeline.exports import RAW_COLUMNS, read_excel_samples, write_excel
from tests.conftest import TMP_DIR
from tests.test_analysis import FPS, moving_track, track


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def excel_name():
    """An Excel file in the same format 8+8+++_fixed.py writes."""
    histories = {
        "1": track(0, [30.0] * 150, [0.0] * 150),
        "2": moving_track(0, 0.0, 150, 20),
        "3": moving_track(10, 45.0, 100, -15),
    }
    histories["3"] = [s.__class__(s.frame_idx, s.t, s.s, s.l, abs(s.speed)) for s in histories["3"]]
    # Vehicle 2 carries raw detection data, as in an Excel file written by this app.
    for i, smp in enumerate(histories["2"]):
        smp.x_px, smp.y_px = 100.0 + i, 500.0
        smp.bbox, smp.confidence, smp.class_raw = (90 + i, 450, 110 + i, 500), 0.8, "two_wheeler"
    classes = {"1": "car", "2": "two_wheeler", "3": "bus"}
    analysis = analyse(histories, classes, FPS, 150, PipelineConfig())
    name = "speed_predictions_test.xlsx"
    write_excel(TMP_DIR / "videos" / name, histories, analysis)
    return name


def wait(client, job_id, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.2)
    raise TimeoutError


def test_excel_roundtrip(excel_name):
    data = read_excel_samples(TMP_DIR / "videos" / excel_name)
    assert set(data["histories"]) == {"1", "2", "3"}
    assert data["fps"] == pytest.approx(FPS, rel=1e-3)
    assert data["classes"]["3"] == "bus"
    first = data["histories"]["2"][0]
    assert (first.x_px, first.bbox, first.confidence) == (100.0, (90, 450, 110, 500), 0.8)
    assert data["histories"]["1"][0].x_px is None  # no raw data -> stays empty


def test_health_and_resources(client, excel_name):
    assert client.get("/health").json()["status"] == "ok"
    assert excel_name in client.get("/excel-results").json()
    assert "road_160000-00094-M" in client.get("/calibrations").json()
    assert client.get("/config/defaults").json()["detection"]["confidence_threshold"] == 0.15


def test_import_job_and_results(client, excel_name):
    r = client.post(
        "/jobs/import",
        json={
            "excel_file": excel_name,
            "config": {"road": {"num_lanes": 2}},
            "location": "Test Rd",
            "recorded_at": "2026-01-30T16:00:00",
        },
    )
    assert r.status_code == 202
    job = wait(client, r.json()["job_id"])
    assert job["status"] == "completed", job["error"]
    assert job["location"] == "Test Rd"
    assert job["recorded_at"].startswith("2026-01-30T16:00:00")
    assert job["run_info"]["source_excel"]["file"] == excel_name
    assert len(job["run_info"]["source_excel"]["sha256"]) == 64
    assert job["metrics"]["vehicles_counted"] == 3
    assert job["metrics"]["parked"] == 1
    assert job["metrics"]["vc_ratio"] is not None

    jid = job["id"]
    vehicles = client.get(f"/jobs/{jid}/vehicles").json()
    assert {v["vehicle_id"]: v["status"] for v in vehicles}["1"] == "Parked"
    assert len(client.get(f"/jobs/{jid}/vehicles", params={"status": "Parked"}).json()) == 1
    samples = client.get(f"/jobs/{jid}/samples", params={"step": 10}).json()
    assert len(samples["t"]) == 15 + 15 + 10
    px_by_vehicle = {}
    for vid, x in zip(samples["vehicle_id"], samples["x_px"], strict=True):
        px_by_vehicle.setdefault(vid, []).append(x)
    assert px_by_vehicle["2"][:2] == [100.0, 110.0]
    assert set(px_by_vehicle["1"]) == {None}

    # Long-format trajectories: one row per vehicle per frame, with raw detection columns.
    r = client.get(f"/jobs/{jid}/files/trajectories_csv")
    assert r.status_code == 200
    traj = pd.read_csv(io.StringIO(r.text))
    assert len(traj) == 150 + 150 + 100
    assert set(RAW_COLUMNS) <= set(traj.columns)
    assert traj.loc[0, "clock_time"] == "2026-01-30T16:00:00.000"
    assert traj.query("vehicle_id == 2")["confidence"].eq(0.8).all()
    parquet = client.get(f"/jobs/{jid}/files/trajectories_parquet")
    assert len(pd.read_parquet(io.BytesIO(parquet.content))) == len(traj)
    assert client.get(f"/jobs/{jid}/files/run_info").json()["location"] == "Test Rd"
    assert client.get(f"/jobs/{jid}/class-summary").status_code == 200
    assert len(client.get(f"/jobs/{jid}/events").json()["stop_events"]) == 1
    assert client.get(f"/jobs/{jid}/files/excel").status_code == 200
    assert client.get(f"/jobs/{jid}/files/video").status_code == 404
    assert client.get(f"/jobs/{jid}/files/secret").status_code == 404

    # Stricter stop rule -> the 150-frame parked car no longer counts as a stop.
    cfg = job["config"]
    cfg["stop_event"]["min_stop_duration_frames"] = 200
    r = client.post(f"/jobs/{jid}/reanalyse", json=cfg)
    assert r.status_code == 200
    assert r.json()["metrics"]["stop_events"] == 0
    assert r.json()["config"]["stop_event"]["min_stop_duration_frames"] == 200
    # Re-analysis keeps the raw detection data and the run info.
    assert r.json()["run_info"]["location"] == "Test Rd"
    again = client.get(f"/jobs/{jid}/samples", params={"vehicle_id": "2"}).json()
    assert again["x_px"][0] == 100.0 and again["confidence"][0] == 0.8

    assert client.delete(f"/jobs/{jid}").status_code == 204
    assert client.get(f"/jobs/{jid}").status_code == 404


def test_rejects_paths_outside_videos_dir(client):
    r = client.post("/jobs/import", json={"excel_file": "../test.db"})
    assert r.status_code == 404
    r = client.post("/jobs", json={"video": "../../etc/passwd", "calibration": "x"})
    assert r.status_code == 404


def test_devices_and_unavailable_device_rejected(client, monkeypatch):
    import torch

    devices = client.get("/devices").json()
    assert devices["cpu"] is True and devices["auto"] in ("0", "mps", "cpu")

    (TMP_DIR / "videos" / "clip.mp4").write_bytes(b"x")  # only needs to exist
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    r = client.post(
        "/jobs",
        json={
            "video": "clip.mp4",
            "calibration": "road_160000-00094-M",
            "config": {"detection": {"device": "mps"}},
        },
    )
    assert r.status_code == 400 and "mps" in r.json()["detail"]
