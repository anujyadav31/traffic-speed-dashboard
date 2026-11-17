"""Submit a video for processing (or import an existing Excel) and watch it live."""

from datetime import datetime

import streamlit as st

import api
from common import fmt, page, require_api

page("Process Video", "🎥")
require_api()

defaults = api.get("/config/defaults")
videos = api.get("/videos")
calibrations = api.get("/calibrations")
devices = api.get("/devices")


def settings_form(prefix: str, full: bool) -> dict:
    """Pipeline settings. Defaults reproduce 8+8+++_fixed.py."""
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in defaults.items()}
    if full:
        with st.expander("Detection & tracking"):
            d = cfg["detection"]
            c = st.columns(4)
            d["confidence_threshold"] = c[0].number_input(
                "Confidence", 0.0, 1.0, float(d["confidence_threshold"]), 0.05, key=f"{prefix}conf"
            )
            d["iou_threshold"] = c[1].number_input(
                "IoU", 0.0, 1.0, float(d["iou_threshold"]), 0.05, key=f"{prefix}iou"
            )
            d["imgsz"] = c[2].selectbox(
                "Image size",
                [640, 960, 1280],
                [640, 960, 1280].index(d["imgsz"]),
                key=f"{prefix}imgsz",
                help="1280 finds small/distant vehicles better but is ~3-4x slower",
            )
            names = {"0": "cuda", "mps": "mps", "cpu": "cpu"}
            options = ["auto"] + [k for k in ("cuda", "mps", "cpu") if devices.get(k)]
            dev = c[3].selectbox(
                "Device",
                options,
                format_func=lambda o: (
                    f"auto ({names.get(devices['auto'], devices['auto'])})" if o == "auto" else o
                ),
                key=f"{prefix}dev",
                help="The whole run uses one device. auto = NVIDIA GPU (cuda) if available, "
                "else Apple GPU (mps), else CPU. Only devices this computer has are listed. "
                "Choose cpu to match 8+8+++_fixed.py exactly (it runs on the CPU).",
            )
            d["device"] = "" if dev == "auto" else dev
        with st.expander("Speed estimation"):
            s = cfg["speed"]
            c = st.columns(3)
            s["time_window_s"] = c[0].number_input(
                "Speed window (s)", 0.2, 5.0, float(s["time_window_s"]), 0.1, key=f"{prefix}tw"
            )
            s["min_movement_m"] = c[1].number_input(
                "Min movement (m)", 0.0, 2.0, float(s["min_movement_m"]), 0.05, key=f"{prefix}mm"
            )
            s["reid_gap_s"] = c[2].number_input(
                "Re-ID gap (s)", 1.0, 30.0, float(s["reid_gap_s"]), 0.5, key=f"{prefix}reid"
            )
            cfg["show_trajectories"] = st.checkbox(
                "Draw trajectory trails", True, key=f"{prefix}trails"
            )
    with st.expander("Stops, queues & road"):
        c = st.columns(3)
        se, q, r = cfg["stop_event"], cfg["queue"], cfg["road"]
        se["max_speed_for_stop_kmh"] = c[0].number_input(
            "Stop speed ≤ (km/h)",
            0.0,
            10.0,
            float(se["max_speed_for_stop_kmh"]),
            0.5,
            key=f"{prefix}ss",
        )
        se["min_stop_duration_frames"] = c[1].number_input(
            "Min stop (frames)", 5, 600, int(se["min_stop_duration_frames"]), 5, key=f"{prefix}sd"
        )
        se["resume_hysteresis_frames"] = c[2].number_input(
            "Resume after (frames)",
            1,
            120,
            int(se["resume_hysteresis_frames"]),
            1,
            key=f"{prefix}rh",
        )
        c = st.columns(3)
        q["queue_speed_threshold_kmh"] = c[0].number_input(
            "Queued speed < (km/h)",
            0.0,
            20.0,
            float(q["queue_speed_threshold_kmh"]),
            0.5,
            key=f"{prefix}qs",
        )
        q["min_queue_duration_frames"] = c[1].number_input(
            "Queued for (frames)", 5, 300, int(q["min_queue_duration_frames"]), 5, key=f"{prefix}qd"
        )
        q["max_queue_distance_m"] = c[2].number_input(
            "Max queue distance (m)",
            5.0,
            500.0,
            float(q["max_queue_distance_m"]),
            5.0,
            key=f"{prefix}qm",
        )
        c = st.columns(3)
        lanes = c[0].number_input("Lanes (0 = skip V/C)", 0, 12, 0, 1, key=f"{prefix}lanes")
        r["num_lanes"] = int(lanes) or None
        r["capacity_per_lane_pcu_hr"] = c[1].number_input(
            "Capacity / lane (PCU/h)",
            100.0,
            3000.0,
            float(r["capacity_per_lane_pcu_hr"]),
            50.0,
            key=f"{prefix}cap",
        )
        r["parked_speed_threshold_kmh"] = c[2].number_input(
            "Parked if median < (km/h)",
            0.0,
            10.0,
            float(r["parked_speed_threshold_kmh"]),
            0.5,
            key=f"{prefix}pk",
        )
    return cfg


def run_details_form(prefix: str) -> dict | None:
    """Optional location and recording start time, saved with the trajectories.

    Returns None (and shows an error) if the time is not valid.
    """
    with st.expander("Location & recording time (optional, saved with the trajectories)"):
        c = st.columns([2, 1, 1])
        location = c[0].text_input(
            "Location / camera", key=f"{prefix}loc", placeholder="e.g. Military Rd, camera 2"
        )
        day = c[1].date_input("Recording date", value=None, key=f"{prefix}day")
        clock = c[2].text_input(
            "Start time (HH:MM:SS)",
            key=f"{prefix}clock",
            placeholder="16:00:00",
            help="Camera clock time of the first frame - adds a real clock time to every "
            "trajectory point",
        )
    recorded_at = None
    if day and clock.strip():
        try:
            recorded_at = datetime.combine(
                day, datetime.strptime(clock.strip(), "%H:%M:%S").time()
            ).isoformat()
        except ValueError:
            st.error("Start time must look like 16:00:00")
            return None
    elif clock.strip() and not day:
        st.error("Pick the recording date as well, or clear the start time.")
        return None
    return {"location": location.strip() or None, "recorded_at": recorded_at}


tab_video, tab_import = st.tabs(["🎥 Process a video", "📥 Import existing Excel results"])

with tab_video:
    if not videos or not calibrations:
        st.warning(
            "No videos or calibrations found. Check VIDEOS_DIR and the calibrations/ folder."
        )
    else:
        c = st.columns([2, 2, 2])
        names = [v["name"] for v in videos]
        default_video = (
            names.index("160000-00094-M_remux.mp4") if ("160000-00094-M_remux.mp4" in names) else 0
        )
        video = c[0].selectbox(
            "Video",
            videos,
            index=default_video,
            format_func=lambda v: f"{v['name']} ({v['size_mb']} MB)",
        )
        calibration = c[1].selectbox("Road calibration", calibrations)
        name = c[2].text_input("Run name", value=video["name"].rsplit(".", 1)[0])
        if video["name"].lower().endswith(".ts"):
            st.warning(
                "`.TS` headers can report the wrong frame rate (15 instead of 14.29 fps), "
                "which overestimates speed by ~5%. Prefer the `_remux.mp4` version."
            )
        c = st.columns([1, 3])
        quick = c[0].checkbox(
            "Quick test", value=True, help="Only process the first N seconds of the video"
        )
        max_seconds = c[1].slider("Seconds to process", 5, 300, 20, 5, disabled=not quick)
        details = run_details_form("v_")
        cfg = settings_form("v_", full=True)
        cfg["max_seconds"] = float(max_seconds) if quick else None
        if st.button("▶️ Start processing", type="primary", disabled=details is None):
            try:
                r = api.post(
                    "/jobs",
                    {
                        "video": video["name"],
                        "calibration": calibration,
                        "name": name,
                        "config": cfg,
                        **details,
                    },
                )
                st.session_state["live_job"] = r["job_id"]
                st.toast(f"Job {r['job_id']} queued")
            except api.ApiError as e:
                st.error(str(e))

with tab_import:
    st.write(
        "Load results already produced by `8+8+++_fixed.py` (or `8+4+++`) – no video "
        "processing. Stops, queues, status and LOS are computed from the per-frame samples."
    )
    excel_files = api.get("/excel-results")
    if not excel_files:
        st.info("No `speed_predictions*.xlsx` files found in the videos folder.")
    else:
        xf = st.selectbox("Excel file", excel_files)
        details_i = run_details_form("i_")
        cfg_i = settings_form("i_", full=False)
        if st.button("📥 Import", type="primary", disabled=details_i is None):
            try:
                r = api.post("/jobs/import", {"excel_file": xf, "config": cfg_i, **details_i})
                st.session_state["live_job"] = r["job_id"]
            except api.ApiError as e:
                st.error(str(e))

st.divider()
st.subheader("Live progress")


@st.fragment(run_every=1.5)
def live_monitor() -> None:
    job_id = st.session_state.get("live_job")
    active = [j for j in api.get("/jobs") if j["status"] in ("queued", "running")]
    if not job_id and active:
        job_id = active[0]["id"]
    if not job_id:
        st.caption("Start a job above to watch it here.")
        return
    job = api.get(f"/jobs/{job_id}")
    status = job["status"]
    stats = job["live_stats"] or {}
    st.write(f"**{job['name']}** · `{job_id}` · status: **{status}**")

    if status in ("queued", "running"):
        total = job["frames_total"] or 0
        st.progress(job["progress"], text=f"Frame {job['frames_done']:,} / {total:,}")
        pfps = stats.get("processing_fps")
        eta = (total - job["frames_done"]) / pfps if pfps and total else None
        c = st.columns(5)
        c[0].metric("Video time", fmt(stats.get("video_time_s"), 1, " s"))
        c[1].metric("Processing speed", fmt(pfps, 1, " fps"))
        c[2].metric("Time left", fmt(eta / 60 if eta else None, 1, " min"))
        c[3].metric("Tracks so far", fmt(stats.get("tracks_so_far"), 0))
        c[4].metric("Mean speed in frame", fmt(stats.get("mean_speed_in_frame"), 1, " km/h"))
        img = api.get_bytes(f"/jobs/{job_id}/preview")
        if img:
            st.image(img, caption="Live annotated frame", width="stretch")
        elif status == "queued":
            st.info("Waiting for a worker… (is `make worker` running?)")
        if st.button("⏹ Cancel", key="cancel_live"):
            api.post(f"/jobs/{job_id}/cancel")
    elif status == "completed":
        st.success("Finished!")
        st.session_state["job_id"] = job_id
        st.page_link("pages/2_Results.py", label="Open results", icon="📊")
    else:
        st.error(f"Job {status}.")
        if job.get("error"):
            st.code(job["error"][:3000])


live_monitor()
