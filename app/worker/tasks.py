"""Celery tasks."""

from app.core.database import init_db
from app.jobs import run_job
from app.worker.celery_app import celery_app


@celery_app.task(name="process_job")
def process_job(job_id: str) -> str:
    init_db()
    run_job(job_id)
    return job_id
