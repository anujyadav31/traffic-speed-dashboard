"""Which device (NVIDIA GPU / Apple GPU / CPU) a run uses - one choice for the whole run.

Kept free of Ultralytics/OpenCV imports so the API can validate a device choice quickly.
"""

import os
from functools import cache


def resolve_device(requested: str) -> str:
    """Device for the whole run. "" (auto) -> NVIDIA GPU, else Apple GPU (mps), else CPU.

    An explicit choice ("cpu", "mps", "cuda"/"0") is checked up front, so a run never
    starts on a device that is not there. Everything torch-based in a run uses this one
    device: the YOLO detector and BoT-SORT's ReID features (Ultralytics hands the
    detector's device to the tracker, and ReID "auto" reuses the detector's own features).
    """
    import torch

    if not requested:
        if torch.cuda.is_available():
            return "0"  # first NVIDIA GPU
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda":
        requested = "0"
    if requested not in ("cpu", "mps") and not requested.isdigit():
        raise ValueError(f"Unknown device '{requested}' - use auto, cuda, mps or cpu")
    if requested.isdigit() and not torch.cuda.is_available():
        raise ValueError("Device 'cuda' was chosen but no NVIDIA GPU is available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Device 'mps' was chosen but the Apple GPU is not available")
    return requested


def mps_cpu_fallback_enabled() -> bool:
    """PYTORCH_ENABLE_MPS_FALLBACK=1 would let PyTorch run unsupported operations on
    the CPU in the middle of an mps run. It is off unless set in the environment."""
    return os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"


@cache
def available_devices() -> dict:
    """What this computer offers, and what "auto" resolves to."""
    import torch

    return {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
        "auto": resolve_device(""),
    }
