"""Edie's generalized traffic flow variables measured in a 3-D time-space shape.

Follows Saberi, Mahmassani, Hou and Zockaie (2014), "Estimating Network Fundamental
Diagram Using Three-Dimensional Vehicle Trajectories", Transportation Research Record
2422, pp. 12-20:

    Q(w) = d(w) / (Lxy(w) * dt)      flow     (eq. 9)
    K(w) = t(w) / (Lxy(w) * dt)      density  (eq. 10)
    V(w) = d(w) / t(w)               speed    (eq. 11)

d(w) is the total distance travelled and t(w) the total time spent by all vehicles inside
the shape w; Lxy(w) is the road length inside w and dt its time height. Here the shape is a
box: distance along the road (s) x sideways offset (l) x time (t). Q = K * V holds by
construction, as the paper shows it must when all three come from the same shape.

Pure pandas/numpy so it can be tested without the dashboard.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Box:
    s0: float  # distance along the road (m)
    s1: float
    l0: float  # sideways offset from the centreline (m)
    l1: float
    t0: float  # time (s)
    t1: float


def smooth_positions(samples: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling median of s and l per vehicle (window in frames). Display/measure option
    only - the stored data and the pipeline's own speeds are not changed."""
    if window <= 1:
        return samples
    out = samples.sort_values(["vehicle_id", "t"]).copy()
    g = out.groupby("vehicle_id", sort=False)
    for col in ("s", "l"):
        if col in out:
            out[col] = g[col].transform(
                lambda x: x.rolling(window, center=True, min_periods=1).median()
            )
    return out


def segments(samples: pd.DataFrame, max_gap_s: float = 1.0) -> pd.DataFrame:
    """Consecutive sample pairs of each vehicle: midpoint (s, l, t), distance and time.

    Pairs further apart than max_gap_s (the vehicle was not detected in between) are
    dropped, so a gap is not counted as distance travelled or time spent.
    """
    df = samples.sort_values(["vehicle_id", "t"])
    nxt = df.groupby("vehicle_id", sort=False).shift(-1)
    seg = pd.DataFrame(
        {
            "vehicle_id": df["vehicle_id"],
            "s": (df["s"] + nxt["s"]) / 2,
            "l": (df["l"] + nxt["l"]) / 2 if "l" in df else np.nan,
            "t": (df["t"] + nxt["t"]) / 2,
            "dist_m": (nxt["s"] - df["s"]).abs(),  # distance along the road
            "dt_s": nxt["t"] - df["t"],
        }
    ).dropna(subset=["dist_m", "dt_s"])
    return seg[(seg["dt_s"] > 0) & (seg["dt_s"] <= max_gap_s)]


def _inside(seg: pd.DataFrame, box: Box, use_lateral: bool) -> pd.DataFrame:
    m = seg["s"].between(box.s0, box.s1) & seg["t"].between(box.t0, box.t1)
    if use_lateral:
        m &= seg["l"].between(box.l0, box.l1)
    return seg[m]


def edie(seg: pd.DataFrame, box: Box, lanes: float = 1.0, use_lateral: bool = True) -> dict:
    """Q, K, V for one box. Units: veh/h (per lane), veh/km (per lane), km/h."""
    inside = _inside(seg, box, use_lateral)
    d_km = inside["dist_m"].sum() / 1000.0
    t_h = inside["dt_s"].sum() / 3600.0
    length_km = max(box.s1 - box.s0, 0.0) / 1000.0
    lane_km = length_km * max(lanes, 1e-9)
    dt_h = max(box.t1 - box.t0, 0.0) / 3600.0
    area = lane_km * dt_h
    q = d_km / area if area > 0 else None
    k = t_h / area if area > 0 else None
    v = d_km / t_h if t_h > 0 else None
    return {
        "vehicles": int(inside["vehicle_id"].nunique()),
        "distance_km": d_km,
        "time_h": t_h,
        "road_length_km": length_km,
        "lane_km": lane_km,
        "dt_h": dt_h,
        "Q_veh_h": q,
        "K_veh_km": k,
        "V_km_h": v,
        "K_times_V": k * v if k is not None and v is not None else None,
    }


def edie_over_time(
    seg: pd.DataFrame, box: Box, interval_s: float, lanes: float = 1.0, use_lateral: bool = True
) -> pd.DataFrame:
    """Q, K, V for consecutive time slices of the box (the paper's NFD points, Fig. 8)."""
    rows = []
    t = box.t0
    while t < box.t1 - 1e-9:
        sub = Box(box.s0, box.s1, box.l0, box.l1, t, min(t + interval_s, box.t1))
        r = edie(seg, sub, lanes, use_lateral)
        rows.append({"t_start_s": sub.t0, "t_end_s": sub.t1, **r})
        t += interval_s
    return pd.DataFrame(rows)
