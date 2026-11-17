"""Key metrics, annotated video and overview charts for one run."""

import plotly.express as px
import streamlit as st

import api
from common import CLASS_COLORS, STATUS_COLORS, metric_strip, page, pick_job, require_api

page("Results", "📊")
require_api()
job = pick_job()
m = job["metrics"] or {}

where_when = " · ".join(
    x
    for x in (
        job.get("location"),
        f"recorded {job['recorded_at'].replace('T', ' ')}" if job.get("recorded_at") else None,
    )
    if x
)
if where_when:
    st.caption(f"📍 {where_when}")

metric_strip(m)
if m.get("vc_ratio") is None:
    st.caption("V/C ratio is empty because no lane count was given for this run.")

vehicles = api.vehicles_df(job["id"])
counted = vehicles[(~vehicles["is_low_sample"]) & (vehicles["vehicle_type"] != "pedestrian")]

left, right = st.columns([3, 2])
with left:
    st.subheader("Annotated video")
    if job["has_video"]:
        st.video(api.video_url(job["id"]))
    else:
        st.info("No video for this run (it was imported from Excel).")
with right:
    st.subheader("Vehicles by type and status")
    if counted.empty:
        st.info("No vehicles.")
    else:
        counts = counted.groupby(["vehicle_type", "status"]).size().reset_index(name="count")
        fig = px.bar(
            counts,
            x="count",
            y="vehicle_type",
            color="status",
            orientation="h",
            color_discrete_map=STATUS_COLORS,
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            legend_title=None,
        )
        st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Speed by vehicle type")
    moving = counted[counted["status"].isin(["Moving", "Stopped"]) & counted["has_speed"]]
    if moving.empty:
        st.info("No moving vehicles.")
    else:
        fig = px.box(
            moving,
            x="vehicle_type",
            y="estimated_speed",
            color="vehicle_type",
            points="all",
            color_discrete_map=CLASS_COLORS,
            hover_data=["vehicle_id", "direction"],
            labels={"estimated_speed": "median speed (km/h)", "vehicle_type": ""},
        )
        fig.update_layout(showlegend=False, height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "One point per vehicle (its median moving speed). Moving and stopped "
            "vehicles only; parked vehicles and short fragments excluded."
        )

with c2:
    st.subheader("Speed over time")
    bin_s = st.select_slider("Time bin (s)", [1, 2, 5, 10, 30, 60], value=5, key="res_bin")
    samples = api.samples_df(job["id"], step=1)
    if samples.empty:
        st.info("No samples.")
    else:
        types = vehicles.set_index("vehicle_id")["vehicle_type"]
        status = vehicles.set_index("vehicle_id")["status"]
        s = samples.assign(
            vehicle_type=samples["vehicle_id"].map(types), status=samples["vehicle_id"].map(status)
        )
        s = s[(s["vehicle_type"] != "pedestrian") & s["status"].isin(["Moving", "Stopped"])]
        s["bin"] = (s["t"] // bin_s) * bin_s
        timeline = (
            s.groupby("bin")
            .agg(
                mean_speed=("speed", lambda x: x[x > 0].mean()),
                vehicles=("vehicle_id", "nunique"),
            )
            .reset_index()
        )
        fig = px.line(
            timeline,
            x="bin",
            y="mean_speed",
            markers=True,
            labels={"bin": "time (s)", "mean_speed": "mean speed (km/h)"},
            hover_data=["vehicles"],
        )
        fig.add_bar(
            x=timeline["bin"],
            y=timeline["vehicles"],
            name="vehicles present",
            yaxis="y2",
            opacity=0.25,
            marker_color="#9aa0a6",
        )
        fig.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis2=dict(overlaying="y", side="right", title="vehicles", showgrid=False),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, width="stretch")

st.subheader("Summary by vehicle type")
cs = api.class_summary_df(job["id"])
st.dataframe(
    cs,
    hide_index=True,
    width="stretch",
    column_config={
        c: st.column_config.NumberColumn(format="%.1f")
        for c in ["mean_speed", "median_speed", "std_speed", "min_speed", "max_speed"]
    },
)
st.caption("Same table as the 'Summary by Class' sheet of 8+8+++_fixed.py (all tracks).")

st.subheader("Downloads")
DOWNLOADS = [
    ("trajectories_csv", "Trajectories (CSV)", "trajectories.csv"),
    ("trajectories_parquet", "Trajectories (Parquet)", "trajectories.parquet"),
    ("excel", "Excel workbook", "speed_predictions.xlsx"),
    ("csv", "Wide CSV (original layout)", "speed_predictions.csv"),
    ("summary", "Summary JSON", "summary.json"),
    ("run_info", "Run info JSON", "run_info.json"),
]
missing = []
for row_start in range(0, len(DOWNLOADS), 3):
    cols = st.columns(3)
    for col, (kind, label, fname) in zip(cols, DOWNLOADS[row_start : row_start + 3], strict=False):
        data = api.file_bytes(job["id"], kind)
        if data:
            col.download_button(label, data, file_name=f"{job['name']}_{fname}", width="stretch")
        else:
            missing.append(label)
st.caption(
    "Trajectories: one row per vehicle per frame – time, road position (`s_m`, `l_m`), speed, "
    "pixel position, bounding box, detection confidence and raw class."
)
if missing:
    st.caption(
        f"Not available for this run: {', '.join(missing)}. Runs made before trajectories "
        "were added get the trajectory files when re-analysed (Events page); pixel data "
        "needs the video to be processed again."
    )

with st.expander("Run details"):
    st.json(
        {
            "id": job["id"],
            "source": job["source"],
            "video": job["video"],
            "calibration": job["calibration"],
            "location": job.get("location"),
            "recorded_at": job.get("recorded_at"),
            "pipeline_version": job["pipeline_version"],
            "config": job["config"],
            "metrics": m,
        }
    )
    if job.get("run_info"):
        st.write("**Run info** – video, calibration copy, model hash, tracker, versions")
        st.json(job["run_info"], expanded=False)
