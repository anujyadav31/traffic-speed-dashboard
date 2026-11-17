"""Application settings, read from environment variables and `.env`."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The original research project sits next to this one. Videos and model weights
# are read from there (read-only) so nothing has to be copied or moved.
SOURCE_PROJECT_DIR = PROJECT_ROOT.parent / "Vehicle detection and speed estimation"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    database_url: str = f"sqlite:///{PROJECT_ROOT / 'traffic.db'}"
    redis_url: str = "redis://localhost:6379/0"
    # "celery" runs jobs on a Celery worker via Redis; "thread" runs them in a
    # background thread inside the API process (handy when Redis is not running).
    job_executor: Literal["celery", "thread"] = "celery"

    videos_dir: Path = SOURCE_PROJECT_DIR
    model_path: Path = SOURCE_PROJECT_DIR / "best_xl_ITD_v1.2.pt"
    calibrations_dir: Path = PROJECT_ROOT / "calibrations"
    runs_dir: Path = PROJECT_ROOT / "runs"

    api_url: str = "http://localhost:8000"
    pipeline_version: str = "1.0.0"


settings = Settings()
