"""3-D space-time trajectories and Edie's flow, density and speed in a 3-D shape.

After Saberi, Mahmassani, Hou and Zockaie (2014), Transportation Research Record 2422:
Fig. 4b (3-D trajectories, x-y = road plane, vertical = time), Fig. 4a (top view),
Fig. 3b (road on the x-y plane, 3-D shape w) and eqs. 9-11 (Edie's definitions in w).
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from edie import Box, edie, edie_over_time, segments, smooth_positions

import api
from common import CLASS_COLORS, fmt, page, pick_job, require_api

page("3-D Space-Time", "🧊")
require_api()
job = pick_job()
vehicles = api.vehicles_df(job["id"])
full = api.samples_df(job["id"], step=1)
if full.empty or vehicles.empty:
    st.info("No trajectories in this run.")
    st.stop()

has_lateral = full["l"].notna().any()
has_pixels = full["x_px"].notna().any()
calib = ((job.get("run_info") or {}).get("calibration") or {}).get("data") or {}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Graph")
    axes_options = ["Road (metres)"] + (["Video image (pixels)"] if has_pixels else [])
    axes = st.radio(
        "Ground plane",
        axes_options,
        help="Road: distance along the calibrated centreline × sideways offset, in metres. "
        "Video image: the pixel position in the camera view.",
    )
    view = st.radio(
        "View",
        ["3-D (Fig. 4b)", "Top – road plane (Fig. 4a)", "Side – distance vs time"],
    )
    color_by = st.radio("Colour by", ["vehicle type", "speed"], horizontal=True)
    step = st.select_slider("Draw every n-th point", [1, 2, 3, 5, 10], value=3)
    types = st.multiselect("Vehicle type", sorted(vehicles["vehicle_type"].unique()))
    direction = st.multiselect("Direction", ["going", "coming", "stationary"])
    hide_low = st.checkbox("Hide low-sample tracks", True)
    hide_ped = st.checkbox("Hide pedestrians", True)

keep = vehicles
if types:
    keep = keep[keep["vehicle_type"].isin(types)]
if direction:
    keep = keep[keep["direction"].isin(direction)]
if hide_low:
    keep = keep[~keep["is_low_sample"]]
if hide_ped:
    keep = keep[keep["vehicle_type"] != "pedestrian"]
info = keep.set_index("vehicle_id")
data = full[full["vehicle_id"].isin(info.index)].copy()
if data.empty:
    st.info("No vehicles match the filters.")
    st.stop()
data["vehicle_type"] = data["vehicle_id"].map(info["vehicle_type"])

s_min, s_max = float(data["s"].min()), float(data["s"].max())
t_min, t_max = float(data["t"].min()), float(data["t"].max())
l_min, l_max = (float(data["l"].min()), float(data["l"].max())) if has_lateral else (0.0, 0.0)

with st.sidebar:
    st.header("Shape ω (Edie)")
    s_rng = st.slider("Distance along road (m)", s_min, s_max, (s_min, s_max), 0.5)
    t_rng = st.slider("Time (s)", t_min, t_max, (t_min, t_max), 0.5)
    use_lat = has_lateral and st.checkbox("Limit sideways offset too", False)
    l_rng = (
        st.slider("Sideways offset (m)", l_min, l_max, (l_min, l_max), 0.25)
        if use_lat
        else (l_min, l_max)
    )
    cfg_lanes = (job.get("config") or {}).get("road", {}).get("num_lanes")
    lanes = st.number_input(
        "Lanes inside ω",
        1,
        12,
        int(cfg_lanes or 1),
        1,
        help="Lxy(ω) = road length × lanes. With 1, flow and density are for the whole road; "
        "with the real lane count they are per lane (as in the paper's lane-miles).",
    )
    smooth = st.select_slider(
        "Reduce position jitter (median over frames)",
        [1, 5, 9, 15],
        value=1,
        help="1 = off (raw positions, as stored). Stopped vehicles near the centreline "
        "bends jump back and forth, which adds distance that was never travelled; a "
        "median over a few frames removes most of it. Affects this page only.",
    )

box = Box(s_rng[0], s_rng[1], l_rng[0], l_rng[1], t_rng[0], t_rng[1])
measured = smooth_positions(data, smooth)
seg = segments(measured)

# ------------------------------------------------------------------ 3-D graph
st.caption(
    "Each line is one vehicle in **two space axes + time** (time is vertical), as in "
    "Saberi et al. (2014) Fig. 4b. Vertical lines = stopped vehicles; slanted lines = moving."
)

plot = measured[measured.groupby("vehicle_id").cumcount() % step == 0]
if axes == "Road (metres)":
    xcol, ycol = "s", "l" if has_lateral else None
    xlab, ylab = "distance along road (m)", "sideways offset (m)"
else:
    xcol, ycol = "x_px", "y_px"
    xlab, ylab = "image x (px)", "image y (px)"
plot = plot.dropna(subset=[xcol] + ([ycol] if ycol else []))
yvals = (lambda d: d[ycol]) if ycol else (lambda d: np.zeros(len(d)))

fig = go.Figure()
if color_by == "vehicle type":
    for cname, grp in plot.groupby("vehicle_type"):
        xs, ys, zs, txt = [], [], [], []
        for vid, v in grp.groupby("vehicle_id", sort=False):
            xs += v[xcol].tolist() + [None]
            ys += list(yvals(v)) + [None]
            zs += v["t"].tolist() + [None]
            txt += [f"#{vid}"] * len(v) + [None]
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name=cname,
                text=txt,
                line=dict(width=3, color=CLASS_COLORS.get(cname)),
                hovertemplate="%{text}<br>x=%{x:.1f}<br>y=%{y:.1f}<br>t=%{z:.1f}s"
                "<extra>" + cname + "</extra>",
            )
        )
else:
    xs, ys, zs, cs, txt = [], [], [], [], []
    for vid, v in plot.groupby("vehicle_id", sort=False):
        xs += v[xcol].tolist() + [None]
        ys += list(yvals(v)) + [None]
        zs += v["t"].tolist() + [None]
        cs += v["speed"].tolist() + [0.0]
        txt += [f"#{vid}"] * len(v) + [None]
    cmax = float(np.nanpercentile(plot["speed"], 98)) or 1.0
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            text=txt,
            showlegend=False,
            line=dict(
                width=3,
                color=cs,
                colorscale="Turbo",
                cmin=0,
                cmax=cmax,
                colorbar=dict(title="km/h"),
            ),
            hovertemplate="%{text}<br>%{line.color:.1f} km/h<br>t=%{z:.1f}s<extra></extra>",
        )
    )

# Road on the ground plane (Fig. 3b): centreline and edges at the start time.
z0 = t_min
if axes == "Road (metres)":
    refs = calib.get("width_refs") or []
    left = max((r.get("left_m", 0) for r in refs), default=0)
    right = max((r.get("right_m", 0) for r in refs), default=0)
    lines = [(0.0, "centreline")]
    if has_lateral and (left or right):
        lines += [(-left, "road edge"), (right, "road edge")]
    for lv, name in lines:
        fig.add_trace(
            go.Scatter3d(
                x=[s_min, s_max],
                y=[lv, lv],
                z=[z0, z0],
                mode="lines",
                line=dict(
                    color="#888",
                    width=4 if name == "centreline" else 2,
                    dash="dash" if name == "road edge" else "solid",
                ),
                name=name,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    # The shape ω as a wireframe box.
    bx = [box.s0, box.s1]
    by = [box.l0, box.l1] if has_lateral else [0.0, 0.0]
    bt = [box.t0, box.t1]
    ex, ey, ez = [], [], []
    for a in (0, 1):
        for b in (0, 1):
            ex += [bx[0], bx[1], None]
            ey += [by[a], by[a], None]
            ez += [bt[b], bt[b], None]
            ex += [bx[a], bx[a], None]
            ey += [by[0], by[1], None]
            ez += [bt[b], bt[b], None]
            ex += [bx[a], bx[a], None]
            ey += [by[b], by[b], None]
            ez += [bt[0], bt[1], None]
    fig.add_trace(
        go.Scatter3d(
            x=ex,
            y=ey,
            z=ez,
            mode="lines",
            name="shape ω",
            line=dict(color="#e83e8c", width=4),  # not a vehicle-type colour
            hoverinfo="skip",
        )
    )
elif calib.get("centreline"):
    cl = np.asarray(calib["centreline"], float)
    fig.add_trace(
        go.Scatter3d(
            x=cl[:, 0],
            y=cl[:, 1],
            z=[z0] * len(cl),
            mode="lines",
            name="centreline",
            line=dict(color="#888", width=5),
            hoverinfo="skip",
        )
    )

cameras = {
    "3-D (Fig. 4b)": dict(eye=dict(x=1.6, y=-1.6, z=0.9), up=dict(x=0, y=0, z=1)),
    "Top – road plane (Fig. 4a)": dict(
        eye=dict(x=0, y=0, z=2.4), up=dict(x=0, y=1, z=0), projection=dict(type="orthographic")
    ),
    "Side – distance vs time": dict(
        eye=dict(x=0, y=-2.4, z=0), up=dict(x=0, y=0, z=1), projection=dict(type="orthographic")
    ),
}
yaxis = dict(title=ylab)
if axes != "Road (metres)":
    yaxis["autorange"] = "reversed"  # image y grows downwards
fig.update_layout(
    height=720,
    margin=dict(l=0, r=0, t=10, b=0),
    legend_title=None,
    uirevision=view + axes,
    scene=dict(
        xaxis=dict(title=xlab),
        yaxis=yaxis,
        zaxis=dict(title="time (s)"),
        aspectmode="manual",
        aspectratio=dict(x=1.6, y=0.9, z=1.0),
        camera=cameras[view],
    ),
)
st.plotly_chart(fig, width="stretch")
if axes != "Road (metres)":
    st.caption("Shape ω is defined in road metres, so it is drawn only in the Road view.")

# ------------------------------------------------------------------ Edie in ω
st.subheader("Edie's flow, density and speed in shape ω")
r = edie(seg, box, lanes, use_lateral=use_lat)
per = " per lane" if lanes > 1 else ""
c = st.columns(5)
c[0].metric(
    f"Flow Q{per} (veh/h)", fmt(r["Q_veh_h"], 0), help="Q(ω) = d(ω) / (Lxy(ω) × Δt) – eq. 9"
)
c[1].metric(
    f"Density K{per} (veh/km)", fmt(r["K_veh_km"], 1), help="K(ω) = t(ω) / (Lxy(ω) × Δt) – eq. 10"
)
c[2].metric("Speed V (km/h)", fmt(r["V_km_h"], 1), help="V(ω) = d(ω) / t(ω) – eq. 11")
c[3].metric(
    "K × V (veh/h)", fmt(r["K_times_V"], 0), help="Equals Q: the fundamental identity Q = K·V"
)
c[4].metric("Vehicles in ω", fmt(r["vehicles"], 0))
st.caption(
    f"d(ω) = {fmt(r['distance_km'] * 1000, 0)} m travelled · t(ω) = {fmt(r['time_h'] * 3600, 0)} "
    f"vehicle-seconds · Lxy(ω) = {fmt(r['road_length_km'] * 1000, 1)} m × {lanes} lane(s) · "
    f"Δt = {fmt(r['dt_h'] * 3600, 1)} s. Distance is measured along the road; gaps where a "
    "vehicle was not detected (> 1 s) are not counted."
)

st.subheader("Flow–density over time (network fundamental diagram points)")
interval = st.select_slider("Time slice (s)", [5, 10, 15, 30, 60], value=10)
ts = edie_over_time(seg, box, interval, lanes, use_lateral=use_lat).dropna(subset=["K_veh_km"])
if ts.empty:
    st.info("No vehicles in ω.")
else:
    c1, c2 = st.columns(2)
    f1 = px.scatter(
        ts,
        x="K_veh_km",
        y="Q_veh_h",
        color="t_start_s",
        color_continuous_scale="Viridis",
        labels={
            "K_veh_km": f"density K{per} (veh/km)",
            "Q_veh_h": f"flow Q{per} (veh/h)",
            "t_start_s": "slice start (s)",
        },
        hover_data=["vehicles", "V_km_h"],
    )
    f1.update_traces(mode="lines+markers", line=dict(color="#9aa0a6", width=1))
    f1.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), title="Q vs K (as Fig. 8)")
    c1.plotly_chart(f1, width="stretch")
    f2 = px.scatter(
        ts,
        x="K_veh_km",
        y="V_km_h",
        color="t_start_s",
        color_continuous_scale="Viridis",
        labels={
            "K_veh_km": f"density K{per} (veh/km)",
            "V_km_h": "speed V (km/h)",
            "t_start_s": "slice start (s)",
        },
        hover_data=["vehicles", "Q_veh_h"],
    )
    f2.update_traces(mode="lines+markers", line=dict(color="#9aa0a6", width=1))
    f2.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), title="V vs K")
    c2.plotly_chart(f2, width="stretch")
    st.dataframe(
        ts[["t_start_s", "t_end_s", "vehicles", "Q_veh_h", "K_veh_km", "V_km_h"]],
        hide_index=True,
        width="stretch",
        column_config={
            k: st.column_config.NumberColumn(format="%.1f")
            for k in ("t_start_s", "t_end_s", "Q_veh_h", "K_veh_km", "V_km_h")
        },
    )
    st.download_button(
        "Download (CSV)",
        ts.to_csv(index=False).encode(),
        file_name=f"{job['name']}_edie_{interval}s.csv",
    )

with st.expander("About this page"):
    st.markdown(
        "- Based on **Saberi, Mahmassani, Hou & Zockaie (2014)**, *Estimating Network "
        "Fundamental Diagram Using Three-Dimensional Vehicle Trajectories*, TRR 2422.\n"
        "- The paper's x-y plane is a city network; here it is one road: **distance along "
        "the calibrated centreline × sideways offset** (or the camera image).\n"
        "- Shape ω is a box (distance × sideways × time). Q, K and V come from the same "
        "shape, so **Q = K·V** holds, as the paper shows.\n"
        "- Positions near sharp bends of the centreline can jump back and forth for stopped "
        "vehicles; use *Reduce position jitter* to see how much that affects Q and V.\n"
        "- Runs imported from Excel have no sideways offset or pixel data, so only "
        "distance × time is shown."
    )
