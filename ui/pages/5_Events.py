"""Stop events, queue events and parked vehicles."""

import pandas as pd
import plotly.express as px
import streamlit as st

import api
from common import CLASS_COLORS, fmt, page, pick_job, require_api

page("Events", "🛑")
require_api()
job = pick_job()
m = job["metrics"] or {}
ev = api.events(job["id"])
stops = pd.DataFrame(ev["stop_events"])
queues = pd.DataFrame(ev["queue_events"])
vehicles = api.vehicles_df(job["id"])

c = st.columns(4)
c[0].metric("Stop events", fmt(m.get("stop_events"), 0))
c[1].metric("Total stopped time", fmt(m.get("total_stop_time_s"), 0, " s"))
c[2].metric("Queue events", fmt(m.get("queue_events"), 0))
c[3].metric("Longest queue", f"{m.get('max_queue_length', 0)} vehicles")

cfg = job["config"]
st.caption(
    f"A stop = speed ≤ {cfg['stop_event']['max_speed_for_stop_kmh']} km/h for at least "
    f"{cfg['stop_event']['min_stop_duration_frames']} frames. A queue = other vehicles behind "
    f"a stopped one, slower than {cfg['queue']['queue_speed_threshold_kmh']} km/h for "
    f"{cfg['queue']['min_queue_duration_frames']} frames, within "
    f"{cfg['queue']['max_queue_distance_m']:.0f} m."
)

with st.expander("🔁 Change thresholds and re-analyse (no need to process the video again)"):
    with st.form("reanalyse"):
        se, q, r = dict(cfg["stop_event"]), dict(cfg["queue"]), dict(cfg["road"])
        c = st.columns(3)
        se["max_speed_for_stop_kmh"] = c[0].number_input(
            "Stop speed ≤ (km/h)", 0.0, 10.0, float(se["max_speed_for_stop_kmh"]), 0.5
        )
        se["min_stop_duration_frames"] = c[1].number_input(
            "Min stop (frames)", 5, 600, int(se["min_stop_duration_frames"]), 5
        )
        se["resume_hysteresis_frames"] = c[2].number_input(
            "Resume after (frames)", 1, 120, int(se["resume_hysteresis_frames"]), 1
        )
        c = st.columns(3)
        q["queue_speed_threshold_kmh"] = c[0].number_input(
            "Queued speed < (km/h)", 0.0, 20.0, float(q["queue_speed_threshold_kmh"]), 0.5
        )
        q["min_queue_duration_frames"] = c[1].number_input(
            "Queued for (frames)", 5, 300, int(q["min_queue_duration_frames"]), 5
        )
        q["max_queue_distance_m"] = c[2].number_input(
            "Max queue distance (m)", 5.0, 500.0, float(q["max_queue_distance_m"]), 5.0
        )
        c = st.columns(4)
        lanes = c[0].number_input("Lanes (0 = skip V/C)", 0, 12, int(r["num_lanes"] or 0), 1)
        r["num_lanes"] = int(lanes) or None
        r["capacity_per_lane_pcu_hr"] = c[1].number_input(
            "Capacity / lane (PCU/h)", 100.0, 3000.0, float(r["capacity_per_lane_pcu_hr"]), 50.0
        )
        r["parked_speed_threshold_kmh"] = c[2].number_input(
            "Parked if median < (km/h)", 0.0, 10.0, float(r["parked_speed_threshold_kmh"]), 0.5
        )
        r["parked_max_displacement_m"] = c[3].number_input(
            "…and moved ≤ (m)", 0.0, 50.0, float(r.get("parked_max_displacement_m", 3.0)), 0.5
        )
        if st.form_submit_button("Re-analyse", type="primary"):
            new_cfg = {**cfg, "stop_event": se, "queue": q, "road": r}
            try:
                api.post(f"/jobs/{job['id']}/reanalyse", new_cfg)
                st.cache_data.clear()
                st.rerun()
            except api.ApiError as e:
                st.error(str(e))

tab1, tab2, tab3 = st.tabs(["Stop events", "Queue events", "Parked vehicles"])

with tab1:
    if stops.empty:
        st.info("No stop events.")
    else:
        types = st.multiselect(
            "Vehicle type", sorted(stops["vehicle_type"].unique()), key="stop_types"
        )
        sf = stops[stops["vehicle_type"].isin(types)] if types else stops
        tl = sf.assign(
            start=pd.to_datetime(sf["start_sec"], unit="s"),
            end=pd.to_datetime(sf["end_sec"], unit="s"),
            vehicle="#" + sf["vehicle_id"].astype(str),
        )
        fig = px.timeline(
            tl,
            x_start="start",
            x_end="end",
            y="vehicle",
            color="vehicle_type",
            color_discrete_map=CLASS_COLORS,
            hover_data=["duration_s", "position_m"],
        )
        fig.update_xaxes(tickformat="%M:%S", title="video time (mm:ss)")
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_layout(
            height=max(300, 18 * len(tl)), margin=dict(l=0, r=0, t=10, b=0), legend_title=None
        )
        st.plotly_chart(fig, width="stretch")
        c1, c2 = st.columns(2)
        fig = px.histogram(
            sf,
            x="position_m",
            nbins=20,
            color="vehicle_type",
            color_discrete_map=CLASS_COLORS,
            labels={"position_m": "stop position along road (m)"},
        )
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            title="Where vehicles stop",
            legend_title=None,
        )
        c1.plotly_chart(fig, width="stretch")
        fig = px.histogram(sf, x="duration_s", nbins=20, labels={"duration_s": "stop duration (s)"})
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), title="How long they stop")
        c2.plotly_chart(fig, width="stretch")
        st.dataframe(sf, hide_index=True, width="stretch")

with tab2:
    if queues.empty:
        st.info("No queue events.")
    else:
        q = queues.assign(
            affected=queues["affected_vehicle_ids"].str.join(", "),
            n_affected=queues["affected_vehicle_ids"].str.len(),
        )
        fig = px.scatter(
            q,
            x="start_sec",
            y="max_queue_length",
            size="n_affected",
            color="blocking_vehicle_type",
            color_discrete_map=CLASS_COLORS,
            hover_data=["blocking_vehicle_id", "duration_s", "affected"],
            labels={
                "start_sec": "queue start (s)",
                "max_queue_length": "max vehicles queued at once",
            },
        )
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend_title=None)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(
            q[
                [
                    "blocking_vehicle_id",
                    "blocking_vehicle_type",
                    "start_sec",
                    "end_sec",
                    "duration_s",
                    "max_queue_length",
                    "max_queue_length_m",
                    "n_affected",
                    "affected",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                c: st.column_config.NumberColumn(format="%.1f")
                for c in ["start_sec", "end_sec", "duration_s", "max_queue_length_m"]
            },
        )
        st.caption(
            "Blocking vehicle = the stopped vehicle; affected = vehicles that were "
            "queued behind it at some point during the stop."
        )

with tab3:
    parked = vehicles[vehicles["status"] == "Parked"]
    if parked.empty:
        st.info("No parked vehicles.")
    else:
        c1, c2 = st.columns([1, 2])
        counts = parked["vehicle_type"].value_counts().reset_index()
        fig = px.pie(
            counts,
            names="vehicle_type",
            values="count",
            hole=0.5,
            color="vehicle_type",
            color_discrete_map=CLASS_COLORS,
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        c1.plotly_chart(fig, width="stretch")
        c2.dataframe(
            parked[
                [
                    "vehicle_id",
                    "vehicle_type",
                    "first_seen_sec",
                    "last_seen_sec",
                    "track_duration_s",
                    "median_raw",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                c: st.column_config.NumberColumn(format="%.1f")
                for c in ["first_seen_sec", "last_seen_sec", "track_duration_s", "median_raw"]
            },
        )
        st.caption(
            "Parked = median speed (including zero samples) below the parked threshold "
            "while visible for at least the minimum stop duration."
        )
