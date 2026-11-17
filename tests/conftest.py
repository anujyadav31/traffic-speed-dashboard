"""Point the app at temporary folders before anything imports the settings."""

import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="tsd_tests_"))
(_tmp / "videos").mkdir()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp / 'test.db'}"
os.environ["RUNS_DIR"] = str(_tmp / "runs")
os.environ["VIDEOS_DIR"] = str(_tmp / "videos")
os.environ["JOB_EXECUTOR"] = "thread"

TMP_DIR = _tmp
