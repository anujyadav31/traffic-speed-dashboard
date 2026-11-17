"""Home: system status and all runs. Start with `make ui`."""

import pandas as pd
import streamlit as st

import api
from common import fmt, local_time, page

page("Traffic Speed Dashboard")
st.caption(
    "Centreline-calibrated vehicle speed estimation (YOLO + BoT-SORT) with stop, queue, "
    "flow and Level-of-Service analysis."
)

h = api.health()
if h is None:
    st.error(f"API not reachable at `{api.API_URL}`. Start it with `make api`.")
    st.stop()

c = st.columns(4)
c[0].metric("API", "Online ✅")
c[1].metric(
    "Job executor",
    h["executor"].capitalize(),
    help="celery = Redis + Celery worker (`make worker`); thread = inside the API",
)
c[2].metric("Model weights", "Found ✅" if h["model_found"] else "Missing ❌")
c[3].metric("Videos folder", "Found ✅" if h["videos_dir_found"] else "Missing ❌")

st.subheader("Runs")
jobs = api.get("/jobs")
if not jobs:
    st.info("No runs yet.")
    st.page_link("pages/1_Process_Video.py", label="Process a video or import results", icon="🎥")
    st.stop()

rows = []
for j in jobs:
    m = j["metrics"] or {}
    rows.append(
        {
            "id": j["id"],
            "name": j["name"],
            "source": j["source"],
            "status": j["status"],
            "progress": j["progress"],
            "created": local_time(j["created_at"]),
            "vehicles": m.get("vehicles_counted"),
            "avg speed (km/h)": m.get("avg_speed_kmh"),
            "LOS": m.get("los_level"),
            "stops": m.get("stop_events"),
            "queues": m.get("queue_events"),
        }
    )
st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    width="stretch",
    column_config={
        "progress": st.column_config.ProgressColumn("progress", min_value=0, max_value=1),
        "avg speed (km/h)": st.column_config.NumberColumn(format="%.1f"),
    },
)

with st.expander("Manage runs"):
    target = st.selectbox(
        "Run", jobs, format_func=lambda j: f"{j['name']} · {j['id']} · {j['status']}"
    )
    col1, col2 = st.columns(2)
    if target["status"] in ("queued", "running") and col1.button("Cancel run"):
        api.post(f"/jobs/{target['id']}/cancel")
        st.rerun()
    if target["status"] not in ("queued", "running") and col2.button("Delete run", type="primary"):
        api.delete(f"/jobs/{target['id']}")
        st.cache_data.clear()
        st.rerun()
    if target.get("error"):
        st.code(target["error"])

st.subheader("Pages")
cols = st.columns(3)
cols[0].page_link("pages/1_Process_Video.py", label="Process Video – run and watch live", icon="🎥")
cols[1].page_link("pages/2_Results.py", label="Results – key metrics, video, charts", icon="📊")
cols[2].page_link("pages/3_Vehicles.py", label="Vehicles – table and per-vehicle speed", icon="🚗")
cols[0].page_link("pages/4_Space_Time.py", label="Space-Time – trajectories and heatmap", icon="🗺️")
cols[1].page_link("pages/5_Events.py", label="Events – stops, queues, parked vehicles", icon="🛑")
cols[2].page_link("pages/6_Compare_Runs.py", label="Compare – runs side by side", icon="⚖️")

latest = next((j for j in jobs if j["status"] == "completed"), None)
if latest:
    st.caption(
        f"Latest completed run: **{latest['name']}** – "
        f"{fmt((latest['metrics'] or {}).get('vehicles_counted'), 0)} vehicles"
    )
