"""Compare two or more runs side by side (e.g. 8+4+++ vs 8+8+++, or two times of day)."""

import pandas as pd
import plotly.express as px
import streamlit as st

import api
from common import job_label, page, require_api

page("Compare Runs", "⚖️")
require_api()

jobs = api.get("/jobs", status="completed")
if len(jobs) < 2:
    st.info("You need at least two completed runs to compare.")
    st.stop()

chosen = st.multiselect(
    "Runs to compare", jobs, default=jobs[:2], format_func=job_label, max_selections=4
)
if len(chosen) < 2:
    st.stop()

METRICS = [
    ("vehicles_counted", "Vehicles counted", "{:.0f}"),
    ("moving", "Moving", "{:.0f}"),
    ("stopped", "Stopped", "{:.0f}"),
    ("parked", "Parked", "{:.0f}"),
    ("low_sample", "Low-sample tracks", "{:.0f}"),
    ("avg_speed_kmh", "Avg operating speed (km/h)", "{:.1f}"),
    ("median_speed_kmh", "Median speed (km/h)", "{:.1f}"),
    ("p85_speed_kmh", "85th pct speed (km/h)", "{:.1f}"),
    ("flow_pcu_hr", "Flow (PCU/h)", "{:.0f}"),
    ("vc_ratio", "V/C ratio", "{:.2f}"),
    ("los_level", "LOS", "{}"),
    ("stop_events", "Stop events", "{:.0f}"),
    ("queue_events", "Queue events", "{:.0f}"),
    ("max_queue_length", "Longest queue", "{:.0f}"),
    ("duration_s", "Video analysed (s)", "{:.1f}"),
]


def short(j: dict) -> str:
    return f"{j['name']} ({j['id'][:6]})"


table = {}
for j in chosen:
    m = j["metrics"] or {}
    table[short(j)] = [fmt.format(m[k]) if m.get(k) is not None else "–" for k, _, fmt in METRICS]
st.subheader("Key metrics")
st.dataframe(pd.DataFrame(table, index=[label for _, label, _ in METRICS]), width="stretch")

frames = []
for j in chosen:
    v = api.vehicles_df(j["id"])
    frames.append(v.assign(run=short(j)))
allv = pd.concat(frames, ignore_index=True)
counted = allv[(~allv["is_low_sample"]) & (allv["vehicle_type"] != "pedestrian")]

c1, c2 = st.columns(2)
with c1:
    st.subheader("Vehicles by type")
    counts = counted.groupby(["run", "vehicle_type"]).size().reset_index(name="count")
    fig = px.bar(counts, x="vehicle_type", y="count", color="run", barmode="group")
    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        legend_title=None,
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, width="stretch")
with c2:
    st.subheader("Speed distribution")
    moving = counted[counted["status"].isin(["Moving", "Stopped"]) & counted["has_speed"]]
    fig = px.box(
        moving,
        x="vehicle_type",
        y="estimated_speed",
        color="run",
        labels={"estimated_speed": "median speed (km/h)", "vehicle_type": ""},
    )
    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        legend_title=None,
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, width="stretch")

st.subheader("Status mix")
mix = counted.groupby(["run", "status"]).size().reset_index(name="count")
fig = px.bar(mix, x="count", y="run", color="status", orientation="h")
fig.update_layout(
    height=80 + 60 * len(chosen),
    margin=dict(l=0, r=0, t=10, b=0),
    legend_title=None,
    yaxis_title=None,
)
st.plotly_chart(fig, width="stretch")
