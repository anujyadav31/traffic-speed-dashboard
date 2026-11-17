"""Celery application. Start a worker with `make worker`."""

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "traffic_speed",
    broker=settings.redis_url,
    backend=settings.redis_url.rsplit("/", 1)[0] + "/1",
    include=["app.worker.tasks"],
)
celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # one video at a time per worker process
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)
