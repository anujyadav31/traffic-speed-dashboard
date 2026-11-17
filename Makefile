.PHONY: help install redis api worker ui dev test lint format clean-runs

help:
	@echo "make install    - install dependencies with uv"
	@echo "make redis      - start Redis (terminal 1)"
	@echo "make api        - start FastAPI on :8000 (terminal 2)"
	@echo "make worker     - start the Celery worker (terminal 3)"
	@echo "make ui         - start the Streamlit dashboard on :8501 (terminal 4)"
	@echo "make dev        - API + dashboard without Redis/Celery (jobs run in the API)"
	@echo "make test       - run tests"
	@echo "make lint       - ruff check"
	@echo "make format     - ruff format"

install:
	uv sync

redis:
	redis-server --port 6379 --save "" --appendonly no

api:
	uv run uvicorn app.api.main:app --reload --port 8000

# -P solo: one video at a time in the main process (PyTorch does not like forked workers).
worker:
	uv run celery -A app.worker.celery_app worker -P solo -c 1 --loglevel=info

ui:
	uv run streamlit run ui/Home.py --server.port 8501

dev:
	JOB_EXECUTOR=thread uv run uvicorn app.api.main:app --port 8000 & \
	uv run streamlit run ui/Home.py --server.port 8501

test:
	uv run pytest -q

lint:
	uv run ruff check app ui tests

format:
	uv run ruff format app ui tests

clean-runs:
	rm -rf runs/* traffic.db traffic.db-shm traffic.db-wal
