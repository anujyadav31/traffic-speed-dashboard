"""Shared page setup, job picker and formatting helpers."""

from datetime import UTC, datetime

import pandas as pd
import streamlit as st

import api

CLASS_COLORS = {
    "two_wheeler": "#2a78d6",
    "autorickshaw": "#e0a106",
    "car": "#1a9e77",
    "bus": "#d9534f",
    "LCV": "#8e6bd6",
    "truck": "#8c5a2b",
    "bicycle": "#17a2b8",
    "pedestrian": "#9aa0a6",
    "unknown": "#5f6368",
}
STATUS_COLORS = {
    "Moving": "#1a9e77",
    "Stopped": "#e0a106",
    "Parked": "#d9534f",
    "Low sample": "#9aa0a6",
    "Pedestrian": "#c3c7cc",
    "Unknown": "#5f6368",
}
STATUS_HELP = {
    "Moving": "Travelled through without a stop event",
    "Stopped": "Had at least one stop event, then moved on",
    "Parked": "Barely moved for its whole visible time",
    "Low sample": "Tracked for fewer than 10 frames (likely a fragment)",
}


def page(title: str, icon: str = "🚦") -> None:
    st.set_page_config(page_title=f"{title} · Traffic Speed", page_icon=icon, layout="wide")
    st.title(f"{icon} {title}")


def require_api() -> None:
    if api.health() is None:
        st.error(f"Cannot reach the API at `{api.API_URL}`. Start it with `make api`.")
        st.stop()


def completed_jobs() -> list[dict]:
    return api.get("/jobs", status="completed")


def local_time(iso: str | None) -> str:
    """API timestamps are UTC; show them in this computer's time zone."""
    if not iso:
        return "–"
    ts = datetime.fromisoformat(iso)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone().strftime("%Y-%m-%d %H:%M")


def job_label(j: dict) -> str:
    when = local_time(j["created_at"])
    src = "📥" if j["source"] == "import" else "🎥"
    return f"{src} {j['name']} · {when} · {j['id']}"


def pick_job(key: str = "job_picker", label: str = "Run") -> dict:
    """Select a completed job; remembers the choice across pages."""
    jobs = completed_jobs()
    if not jobs:
        st.info(
            "No completed runs yet. Process a video or import an Excel file on **Process Video**."
        )
        st.page_link("pages/1_Process_Video.py", label="Go to Process Video", icon="🎥")
        st.stop()
    ids = [j["id"] for j in jobs]
    current = st.session_state.get("job_id")
    index = ids.index(current) if current in ids else 0
    choice = st.sidebar.selectbox(label, jobs, index=index, format_func=job_label, key=key)
    st.session_state["job_id"] = choice["id"]
    return choice


def fmt(x, digits: int = 1, unit: str = "") -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "–"
    return f"{x:,.{digits}f}{unit}"


def metric_strip(m: dict) -> None:
    c = st.columns(6)
    c[0].metric(
        "Vehicles",
        fmt(m.get("vehicles_counted"), 0),
        help="Vehicle tracks with at least 10 samples (pedestrians excluded)",
    )
    c[1].metric("Moving", fmt(m.get("moving"), 0), help=STATUS_HELP["Moving"])
    c[2].metric("Stopped", fmt(m.get("stopped"), 0), help=STATUS_HELP["Stopped"])
    c[3].metric("Parked", fmt(m.get("parked"), 0), help=STATUS_HELP["Parked"])
    c[4].metric(
        "Avg speed (km/h)",
        fmt(m.get("avg_speed_kmh"), 1),
        help="Mean operating speed: mean of per-vehicle median speeds, moving + stopped vehicles",
    )
    c[5].metric("85th pct (km/h)", fmt(m.get("p85_speed_kmh"), 1))
    c = st.columns(6)
    c[0].metric(
        "LOS",
        m.get("los_level") or "–",
        help=f"Level of Service. By speed: {m.get('los_by_speed') or '–'} · "
        f"by V/C: {m.get('los_by_vc') or '–'} (worse one wins)",
    )
    c = c[1:]
    c[0].metric(
        "Flow (PCU/h)",
        fmt(m.get("flow_pcu_hr"), 0),
        help=f"{fmt(m.get('flow_veh_hr'), 0)} vehicles/h – {m.get('vehicles_crossing', '–')} "
        f"vehicles crossed the count line at {fmt(m.get('flow_line_m'), 1, ' m')} along the road",
    )
    c[1].metric(
        "V/C ratio",
        fmt(m.get("vc_ratio"), 2),
        help="Needs the number of lanes (set it when processing)",
    )
    c[2].metric(
        "Stop events",
        fmt(m.get("stop_events"), 0),
        help=f"Total stopped time {fmt(m.get('total_stop_time_s'), 0, ' s')}",
    )
    c[3].metric(
        "Queue events",
        fmt(m.get("queue_events"), 0),
        help=f"Longest queue: {m.get('max_queue_length', 0)} vehicles",
    )
    c[4].metric(
        "Video (s)",
        fmt(m.get("duration_s"), 1),
        help=f"{m.get('frames_processed')} frames at {fmt(m.get('fps'), 2)} fps",
    )
