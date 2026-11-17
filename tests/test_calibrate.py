"""The calibration command, with the OpenCV clicking replaced by fixed answers."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import calibrate
from app.core.config import settings
from app.pipeline.calibrator import RoadCalibrator

CALIB = Path(__file__).resolve().parents[1] / "calibrations" / "road_160000-00094-M.json"


def test_copied_calibrator_reproduces_existing_calibration():
    """Same clicks + same lengths as the saved file -> same cumulative distances."""
    saved = json.loads(CALIB.read_text())
    cal = RoadCalibrator(np.zeros((10, 10, 3), np.uint8))
    cal.centreline = [tuple(p) for p in saved["centreline"]]
    cum = saved["cumulative_dist"]
    seg = [b - a for a, b in zip(cum, cum[1:], strict=False)]
    out = cal.get_calibration_data(saved["total_real_length"], segment_lengths=seg)
    assert out["cumulative_dist"] == pytest.approx(cum)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(5):
        w.write(np.full((240, 320, 3), 128, np.uint8))
    w.release()
    return path


@pytest.fixture
def fake_clicks(monkeypatch, tmp_path):
    """Stand-ins for the OpenCV windows and the terminal prompt."""
    monkeypatch.setattr(settings, "calibrations_dir", tmp_path / "calibrations")

    def centreline(self):
        self.centreline = [(10, 200), (160, 120), (300, 40)]

    monkeypatch.setattr(RoadCalibrator, "run_centreline", centreline)
    monkeypatch.setattr(RoadCalibrator, "run_segment_lengths", lambda self: [None, None])
    monkeypatch.setattr(RoadCalibrator, "run_width_references", lambda self: None)
    monkeypatch.setattr(RoadCalibrator, "run_arbitrary_references", lambda self: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: "30")
    return tmp_path / "calibrations"


def test_command_saves_calibration(video, fake_clicks):
    assert calibrate.main([str(video), "test_road"]) == 0
    data = json.loads((fake_clicks / "test_road.json").read_text())
    assert data["total_real_length"] == 30.0
    assert data["cumulative_dist"][0] == 0.0
    assert data["cumulative_dist"][-1] == pytest.approx(30.0)
    assert len(data["centreline"]) == 3


def test_command_refuses_to_overwrite_without_force(video, fake_clicks):
    assert calibrate.main([str(video), "test_road"]) == 0
    with pytest.raises(SystemExit):
        calibrate.main([str(video), "test_road"])
    assert calibrate.main([str(video), "test_road", "--force"]) == 0


def test_command_rejects_bad_name_and_missing_video(video, fake_clicks):
    with pytest.raises(SystemExit):
        calibrate.main([str(video), "../outside"])
    with pytest.raises(SystemExit):
        calibrate.main(["no_such_video.mp4", "x"])
