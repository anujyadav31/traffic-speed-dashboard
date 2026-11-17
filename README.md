# Traffic Speed Dashboard

A web app built around the speed-estimation pipeline from
`Vehicle detection and speed estimation/8+8+++_fixed.py`:

- **Pipeline:** YOLO (`best_xl_ITD_v1.2.pt`) + BoT-SORT tracking + centreline calibration.
- **Traffic-impedance analysis** (as in `traffic-impedance-analysis-main`): stop events,
  queues, parked vehicles, flow in PCU/h, V/C ratio and Level of Service.
- **Trajectories:** every vehicle, every frame – road position, speed, pixel position,
  bounding box, detection confidence – plus a full record of how each run was made.
- **Backend:** FastAPI + Celery + Redis, with an SQLite database.
- **Dashboard:** multi-page Streamlit with live processing view.

Neither of the two original projects is modified. Videos and model weights are read from
`../Vehicle detection and speed estimation/`, and all outputs are written inside this
folder.

## Architecture

```
Streamlit dashboard (ui/)  ──HTTP──►  FastAPI (app/api)  ──►  Redis queue  ──►  Celery worker
        ▲                                   │                                   │
        └──── polls progress, reads ◄───────┴──── SQLite (traffic.db) ◄─────────┤
              results, plays video                                              │
                                             runs/<job_id>/ (video, Excel, CSV) ◄┘
```

The dashboard only ever talks to the API. That means the same API can later be used
by other clients: scripts, a Next.js frontend, or AI agents.

## Quick start

Needs Python 3.12, [uv](https://docs.astral.sh/uv/), ffmpeg and Redis
(`brew install uv ffmpeg redis`).

```bash
make install        # uv sync
```

### Full setup (Celery + Redis), four terminals

```bash
make redis          # 1. Redis
make api            # 2. FastAPI  -> http://localhost:8000/docs
make worker         # 3. Celery worker (processes the videos)
make ui             # 4. Dashboard -> http://localhost:8501
```

### Quick try without Redis

```bash
make dev            # jobs run in a thread inside the API process
```

## Dashboard pages

| Page | What it shows |
|---|---|
| **Home** | System status and all runs, with cancel and delete |
| **Process Video** | Pick a video, calibration and settings, optionally the location and recording start time, start a run, and watch the annotated frames, progress, speed and ETA live. Can also **import** Excel files already produced by `8+8+++_fixed.py` / `8+4+++` |
| **Results** | Key metrics (vehicles, moving/stopped/parked, speeds, LOS, flow, V/C, stops, queues), the annotated video, vehicles by type and status, speed by type, speed over time, downloads (trajectories CSV/Parquet, Excel, CSV, JSON) and the full run info |
| **Vehicles** | Filterable table. Click a vehicle to see its speed and position over time, its path on the road, and all statistics |
| **Space-Time** | Space-time diagram (by type or speed), speed heatmap (time × distance), top view of the road with stop locations, and the saved trajectories drawn on a frame of the video |
| **Events** | Stop-event timeline, stop positions and durations, queue events, parked vehicles. Change the thresholds and **re-analyse** without reprocessing the video |
| **Compare Runs** | Up to four runs side by side (e.g. `8+4+++` vs `8+8+++`, or two times of day) |

## What is the same as `8+8+++_fixed.py`

- `RoadMapper` / `segment_intersection` (`app/pipeline/geometry.py`) and the drawing
  helpers (`app/pipeline/drawing.py`) are **copied verbatim**.
- The per-frame loop in `app/pipeline/speed_pipeline.py` uses the same logic and default
  settings:
  - confidence 0.15, IoU 0.5, image size 640
  - BoT-SORT with re-identification (ReID) and the same parameters
  - pedestrian suppression
  - class smoothing over 10 frames
  - 5 s re-ID gap
  - 1 s speed window, 0.2 m minimum movement
- The Excel file keeps the three original sheets and formulas ("Per Frame Samples",
  "Per Vehicle Summary", "Summary by Class"). The original columns come first, in the
  original order. Added: `lateral_m` and the raw detection columns in the per-frame
  sheet, `status`, `stopped_time_s` and `pcu` in the vehicle sheet, and four new sheets:
  "Stop Events", "Queue Events", "Run Metrics" and "Run Info".

## Differences from `8+8+++_fixed.py`

The differences are only what running as a background job requires:

- No OpenCV windows or key presses. The latest frame is written to `preview.jpg` for
  the live view instead.
- Each run writes to its own folder: `runs/<job_id>/`.
- The video is re-encoded to H.264 so that browsers can play it.
- There is an optional "process only the first N seconds" limit, and runs can be
  cancelled.
- The pixel point, bounding box, detection confidence and unsmoothed class of every
  detection are *recorded* (see Trajectories). They are not used in any calculation.
- The SSL certificate bypass at the top of the script is not included (the model file
  is already local, so nothing is downloaded).
- Device "auto" uses an NVIDIA GPU if there is one, else the Apple GPU (`mps`), else the
  CPU. `8+8+++_fixed.py` sets no device, so it runs on the CPU on a Mac; choose
  **Device: cpu** to match it exactly (results on a GPU can differ very slightly).
- The installed Ultralytics version (8.4.x) may be newer than the one used for the
  original results. That can change detections and tracking slightly.

## Trajectories

Every run saves, for every vehicle in every frame where it was detected:

| Group | Fields |
|---|---|
| Identity and time | `run_id`, `vehicle_id`, `vehicle_type`, `direction`, `status`, `frame_idx`, `timestamp_s`, `clock_time` (if a recording start time was entered) |
| Road position | `s_m` (distance along the calibrated centreline), `l_m` (sideways offset), `speed_kmh` |
| Raw image data | `x_px`, `y_px` (bottom-centre of the box: the point used for speed), `bbox_x1_px` … `bbox_y2_px`, `confidence`, `class_raw` (this frame's class before smoothing) |

Keeping the raw pixel data means the road positions can be recomputed later with a
better calibration, without processing the video again.

Each run also stores **run info** (`run_info.json`, the "Run Info" Excel sheet and the
database): location and recording time, video file details (size, fps, frame count,
resolution), a **full copy of the calibration**, the model file and its SHA-256 hash,
detection, speed and tracker settings, and software versions.

Files in `runs/<job_id>/`: `trajectories.csv`, `trajectories.parquet`, `run_info.json`,
`speed_predictions.xlsx`, `speed_predictions.csv`, `summary.json`, `annotated.mp4`
and `frame.jpg` (a clean first frame, used to draw trajectories on the video).

Only real detections are stored; gaps where a vehicle was not detected are left as
gaps, not filled in. Runs imported from Excel, and runs made before this was added,
have no pixel data; re-analysing them (Events page) creates their trajectory files.

The database upgrades itself: when the app starts, new columns are added to an
existing `traffic.db`, and existing runs are kept.

## Traffic analysis (added)

| Output | Definition |
|---|---|
| Stop event | Speed ≤ 2 km/h for ≥ 45 frames. It ends after 10 frames above that speed. Same state machine as `traffic-impedance-analysis-main`. Pedestrians are ignored |
| Queue event | While a vehicle is stopped: other vehicles *behind* it, slower than 2 km/h for 30 frames, within 60 m. "Behind" follows each vehicle's direction of travel (going/coming) and, when lateral positions are known, the same side of the road |
| Parked | Median speed (including zero samples) < 2 km/h, visible for ≥ 45 frames, and moved ≤ 3 m in total |
| Stopped | At least one stop event, but not parked |
| Low sample | Fewer than 10 samples (probably a track fragment); excluded from metrics |
| Flow | Moving and stopped vehicles crossing a count line across the road (by default the middle of the observed section), converted to vehicles/h and PCU/h |
| PCU | car 1, two-wheeler 0.5, bicycle 0.5, autorickshaw 1.2, LCV 1.4, bus 3, truck 3 |
| LOS | Speed thresholds and V/C thresholds from `traffic-impedance-analysis-main` (HCM/IRC); the worse of the two is used. V/C needs the number of lanes |

All thresholds can be changed in the dashboard.

## API

Interactive documentation is at <http://localhost:8000/docs>. Main endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/videos`, `/calibrations`, `/excel-results`, `/config/defaults` | Available inputs |
| POST | `/jobs` | Process a video `{video, calibration, name?, location?, recorded_at?, config?}` |
| POST | `/jobs/import` | Import an Excel file from the original scripts (same optional `location`, `recorded_at`) |
| GET | `/jobs`, `/jobs/{id}` | Status, progress, live stats, metrics |
| GET | `/jobs/{id}/preview` | Latest annotated frame while running |
| POST | `/jobs/{id}/cancel`, `/jobs/{id}/reanalyse` · DELETE `/jobs/{id}` | Manage runs |
| GET | `/jobs/{id}/vehicles`, `/samples`, `/events`, `/class-summary` | Results |
| GET | `/jobs/{id}/files/{kind}` | Downloads: `trajectories_csv`, `trajectories_parquet`, `run_info`, `excel`, `csv`, `summary`, `video`, `frame` |

## Adding a road calibration

Calibrations are the `road_calibration.json` files written by the click-based
calibration tool in `8+8+++_fixed.py`. To add one, copy it into `calibrations/` under a
descriptive name (for example `calibrations/military_road_cam2.json`). The file name
becomes its name in the dashboard. `road_160000-00094-M.json` is a copy of the
calibration used by `8+8+++_fixed.py`.

## Project layout

```
app/
  api/            FastAPI app and routes
  core/           settings and database
  pipeline/       geometry.py, drawing.py (verbatim), speed_pipeline.py, analysis.py,
                  exports.py, config.py
  worker/         Celery app and task
  jobs.py         runs a job, stores results, re-analysis
  models.py       SQLAlchemy tables
  schemas.py      Pydantic request/response models
ui/               Streamlit dashboard (Home.py + pages/)
tests/            pytest: geometry, analysis, API, database upgrade
calibrations/     road calibration JSON files
runs/             outputs, one folder per run
```

## Tests

```bash
make test
make lint
```

## Speed on a Mac

Device "auto" (the default) uses the Apple GPU (`mps`) on a Mac. It measured about
8 frames/s, so a 90-second clip takes about 3 minutes. On the CPU (**Device: cpu**) it
measured about 4 frames/s. Each run's info records the device actually used. Image
size 1280 finds more small or distant vehicles, but is several times slower.
