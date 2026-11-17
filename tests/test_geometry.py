import json
from pathlib import Path

import pytest

from app.pipeline.geometry import RoadMapper

CALIB = Path(__file__).resolve().parents[1] / "calibrations" / "road_160000-00094-M.json"


@pytest.fixture(scope="module")
def mapper():
    return RoadMapper(json.loads(CALIB.read_text()))


def test_centreline_points_map_to_cumulative_distance(mapper):
    calib = json.loads(CALIB.read_text())
    for pt, dist in zip(calib["centreline"], calib["cumulative_dist"], strict=True):
        s, lat = mapper.image_to_world(pt)
        assert s == pytest.approx(dist, abs=1e-3)
        assert lat == pytest.approx(0.0, abs=1e-6)


def test_distance_along_straight_segment(mapper):
    # Two points on the first straight segment, 1/4 and 3/4 of the way along it.
    (x0, y0), (x1, y1) = mapper.points[0], mapper.points[1]
    a = mapper.image_to_world((x0 + 0.25 * (x1 - x0), y0 + 0.25 * (y1 - y0)))
    b = mapper.image_to_world((x0 + 0.75 * (x1 - x0), y0 + 0.75 * (y1 - y0)))
    seg_len = mapper.cum_dist[1] - mapper.cum_dist[0]
    assert mapper.physical_distance(a, b) == pytest.approx(0.5 * seg_len, rel=1e-3)


def test_lateral_offset_has_opposite_signs_on_each_side(mapper):
    (x0, y0), (x1, y1) = mapper.points[1], mapper.points[2]
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    nx, ny = -(y1 - y0), (x1 - x0)  # normal to the segment
    norm = (nx**2 + ny**2) ** 0.5
    _, left = mapper.image_to_world((mx + 50 * nx / norm, my + 50 * ny / norm))
    _, right = mapper.image_to_world((mx - 50 * nx / norm, my - 50 * ny / norm))
    assert left * right < 0


def test_auto_device_prefers_nvidia_then_mps_then_cpu(monkeypatch):
    import torch

    from app.pipeline.device import resolve_device

    def set_hw(cuda: bool, mps: bool):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    set_hw(cuda=True, mps=True)
    assert resolve_device("") == "0"
    set_hw(cuda=False, mps=True)
    assert resolve_device("") == "mps"
    set_hw(cuda=False, mps=False)
    assert resolve_device("") == "cpu"
    assert resolve_device("cpu") == "cpu"  # an explicit choice is kept


def test_explicit_device_is_checked(monkeypatch):
    import pytest
    import torch

    from app.pipeline.device import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    for bad in ("cuda", "0", "mps", "gpu"):
        with pytest.raises(ValueError):
            resolve_device(bad)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == "0"
