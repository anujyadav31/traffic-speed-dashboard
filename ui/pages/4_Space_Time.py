"""Space-time diagram, speed heatmap, top view of the road and trajectories on the video."""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import api
from common import CLASS_COLORS, page, pick_job, require_api

page("Space-Time", "🗺️")
require_api()
job = pick_job()

vehicles = api.vehicles_df(job["id"])
with st.sidebar:
    st.header("Options")
    step = st.select_slider(
        "Use every n-th sample",
        [1, 2, 3, 5, 10],
        value=2,
        help="Higher = faster charts, less detail",
    )
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

samples = api.samples_df(job["id"], step=step)
s = samples[samples["vehicle_id"].isin(info.index)].copy()
if s.empty:
    st.info("No samples match the filters.")
    st.stop()
s["vehicle_type"] = s["vehicle_id"].map(info["vehicle_type"])
s["status"] = s["vehicle_id"].map(info["status"])

tab1, tab2, tab3, tab4 = st.tabs(
    ["Space-time diagram", "Speed heatmap", "Top view", "Video view (pixels)"]
)

with tab4:
    pix = s.dropna(subset=["x_px", "y_px"])
    if pix.empty:
        st.info(
            "Pixel positions are not stored for this run – it was imported from Excel or "
            "processed before trajectories were saved. Process the video again to see them."
        )
    else:
        color_by_px = st.radio("Colour by", ["vehicle type", "speed"], horizontal=True, key="pxc")
        frame = api.file_bytes(job["id"], "frame")
        if frame:
            img = Image.open(io.BytesIO(frame)).convert("RGB")
            full_w, full_h = img.size
            small = img.resize((full_w // 2, full_h // 2))  # lighter page, same coordinates
            fig = px.imshow(
                np.asarray(small),
                x=np.arange(small.width) * full_w / small.width,
                y=np.arange(small.height) * full_h / small.height,
            )
            fig.update_traces(hoverinfo="skip", hovertemplate=None)
        else:
            fig = go.Figure()
            fig.update_yaxes(autorange="reversed")
        if color_by_px == "vehicle type":
            for cname, grp in pix.groupby("vehicle_type"):
                xs, ys, texts = [], [], []
                for vid, v in grp.groupby("vehicle_id", sort=False):
                    xs += v["x_px"].tolist() + [None]
                    ys += v["y_px"].tolist() + [None]
                    texts += [f"#{vid}"] * len(v) + [None]
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        name=cname,
                        text=texts,
                        line=dict(width=2, color=CLASS_COLORS.get(cname)),
                        hovertemplate="%{text}<extra>" + cname + "</extra>",
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=pix["x_px"],
                    y=pix["y_px"],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=pix["speed"],
                        colorscale="Turbo",
                        cmin=0,
                        cmax=float(np.nanpercentile(pix["speed"], 98)) or 1,
                        colorbar=dict(title="km/h"),
                    ),
                    text="#" + pix["vehicle_id"].astype(str),
                    hovertemplate="%{text}<br>%{marker.color:.1f} km/h<extra></extra>",
                    showlegend=False,
                )
            )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0), legend_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Saved trajectories (the bottom-centre point of each box, the point used for "
            "speed) drawn on the first frame of the video."
        )

with tab1:
    color_by = st.radio("Colour by", ["vehicle type", "speed"], horizontal=True)
    fig = go.Figure()
    if color_by == "vehicle type":
        # One trace per class; vehicles separated by None so each is its own line.
        for cname, grp in s.groupby("vehicle_type"):
            xs, ys, texts = [], [], []
            for vid, v in grp.groupby("vehicle_id", sort=False):
                xs += v["t"].tolist() + [None]
                ys += v["s"].tolist() + [None]
                texts += [f"#{vid}"] * len(v) + [None]
            fig.add_trace(
                go.Scattergl(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name=cname,
                    text=texts,
                    line=dict(width=1.5, color=CLASS_COLORS.get(cname)),
                    hovertemplate="%{text}<br>t=%{x:.1f}s<br>"
                    "s=%{y:.1f}m<extra>" + cname + "</extra>",
                )
            )
    else:
        fig.add_trace(
            go.Scattergl(
                x=s["t"],
                y=s["s"],
                mode="markers",
                marker=dict(
                    size=3,
                    color=s["speed"],
                    colorscale="Turbo",
                    cmin=0,
                    cmax=float(np.nanpercentile(s["speed"], 98)) or 1,
                    colorbar=dict(title="km/h"),
                ),
                text="#" + s["vehicle_id"].astype(str),
                hovertemplate="%{text}<br>t=%{x:.1f}s<br>s=%{y:.1f}m<br>"
                "%{marker.color:.1f} km/h<extra></extra>",
            )
        )
    fig.update_layout(
        height=560,
        xaxis_title="time (s)",
        yaxis_title="distance along road (m)",
        margin=dict(l=0, r=0, t=10, b=0),
        legend_title=None,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Each line is one vehicle. Flat = stopped or parked; steep = fast. Lines going "
        "up travel along the calibrated centreline ('going'), lines going down travel "
        "against it ('coming'). Queues show as bundles of flattening lines behind a "
        "flat one."
    )

with tab2:
    c = st.columns(2)
    t_bin = c[0].select_slider("Time bin (s)", [1, 2, 5, 10, 30], value=5)
    s_bin = c[1].select_slider("Distance bin (m)", [1, 2, 5, 10], value=5)
    moving = s.copy()
    moving["t_bin"] = (moving["t"] // t_bin) * t_bin
    moving["s_bin"] = (moving["s"] // s_bin) * s_bin
    grid = moving.groupby(["s_bin", "t_bin"])["speed"].mean().reset_index()
    pivot = grid.pivot(index="s_bin", columns="t_bin", values="speed")
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="RdYlGn",
            zmin=0,
            zmax=float(np.nanpercentile(grid["speed"], 95)) or 1,
            colorbar=dict(title="km/h"),
            hovertemplate="t=%{x}s<br>s=%{y}m<br>%{z:.1f} km/h<extra></extra>",
        )
    )
    fig.update_layout(
        height=520,
        xaxis_title="time (s)",
        yaxis_title="distance along road (m)",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Mean speed of all samples in each cell (including stationary ones). Red "
        "areas that persist over time are where traffic is held up."
    )

with tab3:
    if s["l"].isna().all():
        st.info(
            "Lateral positions are not available for imported runs – process the video "
            "in this app to get the top view."
        )
    else:
        top = s.dropna(subset=["l"])
        mode = st.radio("Show", ["average speed per cell", "all points"], horizontal=True)
        if mode == "all points":
            fig = px.scatter(
                top,
                x="s",
                y="l",
                color="speed",
                color_continuous_scale="Turbo",
                range_color=(0, float(np.nanpercentile(top["speed"], 98)) or 1),
                hover_data=["vehicle_id", "vehicle_type"],
                opacity=0.5,
            )
            fig.update_traces(marker_size=3)
        else:
            top = top.assign(sb=(top["s"] // 2) * 2, lb=(top["l"] // 0.5) * 0.5)
            cells = top.groupby(["sb", "lb"])["speed"].mean().reset_index()
            fig = px.density_heatmap(
                cells,
                x="sb",
                y="lb",
                z="speed",
                histfunc="avg",
                nbinsx=int(np.ptp(cells["sb"]) // 2 + 1),
                nbinsy=int(np.ptp(cells["lb"]) // 0.5 + 1),
                color_continuous_scale="RdYlGn",
            )
        fig.update_layout(
            height=460,
            xaxis_title="distance along road (m)",
            yaxis_title="lateral offset from centreline (m)",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, width="stretch")
        stops = pd.DataFrame(api.events(job["id"])["stop_events"])
        if not stops.empty and stops["lateral_m"].notna().any():
            st.caption("Stop locations:")
            fig = px.scatter(
                stops.dropna(subset=["lateral_m"]),
                x="position_m",
                y="lateral_m",
                size="duration_s",
                color="vehicle_type",
                color_discrete_map=CLASS_COLORS,
                hover_data=["vehicle_id"],
            )
            fig.update_layout(
                height=320,
                xaxis_title="distance along road (m)",
                yaxis_title="lateral offset (m)",
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, width="stretch")
