"""Filterable vehicle table and per-vehicle speed / position plots."""

import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

import api
from common import STATUS_HELP, page, pick_job, require_api

page("Vehicles", "🚗")
require_api()
job = pick_job()
df = api.vehicles_df(job["id"])
if df.empty:
    st.info("No vehicles in this run.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    types = st.multiselect("Vehicle type", sorted(df["vehicle_type"].unique()))
    statuses = st.multiselect(
        "Status",
        sorted(df["status"].unique()),
        help="\n".join(f"- **{k}**: {v}" for k, v in STATUS_HELP.items()),
    )
    directions = st.multiselect("Direction", sorted(df["direction"].unique()))
    hide_low = st.checkbox("Hide low-sample tracks", True)
    hide_ped = st.checkbox("Hide pedestrians", True)
    top = max(1.0, float(round(float(df["estimated_speed"].max() or 0) + 1)))
    speed_range = st.slider("Median speed (km/h)", 0.0, top, (0.0, top))

f = df.copy()
if types:
    f = f[f["vehicle_type"].isin(types)]
if statuses:
    f = f[f["status"].isin(statuses)]
if directions:
    f = f[f["direction"].isin(directions)]
if hide_low:
    f = f[~f["is_low_sample"]]
if hide_ped:
    f = f[f["vehicle_type"] != "pedestrian"]
sp = f["estimated_speed"].fillna(0)
f = f[(sp >= speed_range[0]) & (sp <= speed_range[1])]

st.write(f"**{len(f)}** of {len(df)} tracks")
cols = [
    "vehicle_id",
    "vehicle_type",
    "status",
    "direction",
    "estimated_speed",
    "p85",
    "max_speed",
    "first_seen_sec",
    "track_duration_s",
    "stopped_time_s",
    "n_flow_samples",
    "pcu",
]
event = st.dataframe(
    f[cols],
    hide_index=True,
    width="stretch",
    height=380,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "estimated_speed": st.column_config.NumberColumn("median speed", format="%.1f"),
        "p85": st.column_config.NumberColumn("p85", format="%.1f"),
        "max_speed": st.column_config.NumberColumn("max", format="%.1f"),
        "first_seen_sec": st.column_config.NumberColumn("first seen (s)", format="%.1f"),
        "track_duration_s": st.column_config.NumberColumn("duration (s)", format="%.1f"),
        "stopped_time_s": st.column_config.NumberColumn("stopped (s)", format="%.1f"),
    },
)
st.download_button(
    "Download filtered table (CSV)",
    f.to_csv(index=False).encode(),
    file_name=f"{job['name']}_vehicles.csv",
)

st.subheader("Vehicle detail")
rows = event.selection.rows if event and event.selection else []
if rows:
    vid = f.iloc[rows[0]]["vehicle_id"]
else:
    vid = st.selectbox("Or pick a vehicle", f["vehicle_id"].tolist()) if len(f) else None
if vid is None:
    st.stop()

samples = api.samples_df(job["id"], step=1)
v = samples[samples["vehicle_id"] == vid]
info = df.set_index("vehicle_id").loc[vid]
st.write(
    f"**#{vid}** · {info['vehicle_type']} · {info['status']} · {info['direction']} · "
    f"median {info['estimated_speed'] or 0:.1f} km/h"
)

fig = make_subplots(
    rows=1, cols=2, subplot_titles=("Speed over time", "Distance along road over time")
)
fig.add_scatter(x=v["t"], y=v["speed"], mode="lines", name="speed (km/h)", row=1, col=1)
fig.add_scatter(x=v["t"], y=v["s"], mode="lines", name="position (m)", row=1, col=2)
fig.update_xaxes(title_text="time (s)")
fig.update_yaxes(title_text="km/h", row=1, col=1)
fig.update_yaxes(title_text="m", row=1, col=2)
fig.update_layout(height=320, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, width="stretch")

if v["l"].notna().any():
    fig = px.scatter(
        v,
        x="s",
        y="l",
        color="speed",
        color_continuous_scale="Turbo",
        labels={"s": "distance along road (m)", "l": "lateral offset (m)"},
        title="Path on the road (top view)",
    )
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")

with st.expander("All statistics (same as 'Per Vehicle Summary')"):
    st.json(api.vehicle_stats(job["id"], vid))
