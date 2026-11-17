"""FastAPI application. Run with `make api` (Swagger UI at /docs)."""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.api.jobs import router as jobs_router
from app.core.config import settings
from app.core.database import init_db
from app.pipeline.config import PipelineConfig
from app.pipeline.device import available_devices

VIDEO_EXTENSIONS = {".mp4", ".ts", ".mov", ".avi", ".mkv"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    yield


app = FastAPI(
    title="Traffic Speed Dashboard API", version=settings.pipeline_version, lifespan=lifespan
)
app.include_router(jobs_router)


@app.get("/health", tags=["meta"])
def health():
    return {
        "status": "ok",
        "executor": settings.job_executor,
        "model_found": settings.model_path.is_file(),
        "videos_dir_found": settings.videos_dir.is_dir(),
    }


@app.get("/devices", tags=["meta"])
def devices() -> dict:
    """Processing devices on this computer and what "auto" picks
    (NVIDIA GPU, else Apple GPU / mps, else CPU)."""
    return available_devices()


@app.get("/videos", tags=["resources"])
def list_videos() -> list[dict]:
    if not settings.videos_dir.is_dir():
        return []
    return [
        {"name": p.name, "size_mb": round(p.stat().st_size / 1e6, 1)}
        for p in sorted(settings.videos_dir.iterdir())
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
        and not p.name.startswith("output_annotated")
    ]


@app.get("/excel-results", tags=["resources"])
def list_excel_results() -> list[str]:
    """Excel outputs of the original scripts that can be imported as jobs."""
    if not settings.videos_dir.is_dir():
        return []
    return sorted(p.name for p in settings.videos_dir.glob("speed_predictions*.xlsx"))


@app.get("/calibrations", tags=["resources"])
def list_calibrations() -> list[str]:
    return sorted(p.stem for p in settings.calibrations_dir.glob("*.json"))


@app.get("/calibrations/{name}", tags=["resources"])
def get_calibration(name: str) -> dict:
    path = settings.calibrations_dir / f"{name}.json"
    if path.parent.resolve() != settings.calibrations_dir.resolve() or not path.is_file():
        raise HTTPException(404, "Calibration not found")
    return json.loads(path.read_text())


@app.get("/config/defaults", tags=["resources"])
def default_config() -> dict:
    return PipelineConfig().model_dump()
