"""Thin client for the FastAPI backend. The dashboard never touches the pipeline directly."""

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")


class ApiError(Exception):
    pass


def _req(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=kw.pop("timeout", 30), **kw)
    except requests.ConnectionError as exc:
        raise ApiError(f"Cannot reach the API at {API_URL}. Is it running? (`make api`)") from exc
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ApiError(f"{r.status_code}: {detail}")
    return r


def get(path: str, **params):
    return _req("GET", path, params=params).json()


def post(path: str, json: dict | None = None):
    return _req("POST", path, json=json).json()


def delete(path: str) -> None:
    _req("DELETE", path)


def get_bytes(path: str) -> bytes | None:
    try:
        return _req("GET", path, timeout=10).content
    except ApiError:
        return None


def health() -> dict | None:
    try:
        return get("/health")
    except ApiError:
        return None


# ---- cached reads of finished jobs (results never change once completed) ----
@st.cache_data(ttl=3600, show_spinner=False)
def vehicles_df(job_id: str) -> pd.DataFrame:
    return pd.DataFrame(get(f"/jobs/{job_id}/vehicles"))


@st.cache_data(ttl=3600, show_spinner=False)
def class_summary_df(job_id: str) -> pd.DataFrame:
    return pd.DataFrame(get(f"/jobs/{job_id}/class-summary"))


@st.cache_data(ttl=3600, show_spinner="Loading samples...")
def samples_df(job_id: str, step: int = 1) -> pd.DataFrame:
    return pd.DataFrame(get(f"/jobs/{job_id}/samples", step=step))


@st.cache_data(ttl=3600, show_spinner=False)
def events(job_id: str) -> dict:
    return get(f"/jobs/{job_id}/events")


@st.cache_data(ttl=3600, show_spinner=False)
def vehicle_stats(job_id: str, vehicle_id: str) -> dict:
    return get(f"/jobs/{job_id}/vehicles/{vehicle_id}/stats")


@st.cache_data(ttl=3600, show_spinner=False)
def file_bytes(job_id: str, kind: str) -> bytes | None:
    return get_bytes(f"/jobs/{job_id}/files/{kind}")


def video_url(job_id: str) -> str:
    return f"{API_URL}/jobs/{job_id}/files/video"
