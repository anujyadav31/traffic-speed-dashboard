"""Create a road calibration with OpenCV windows, exactly like 8+8+++_fixed.py does.

Run it in a terminal (it opens windows on your screen and asks for distances):

    uv run python -m app.calibrate 160000-00094-M_remux.mp4 military_road_cam2

The steps are the same as in 8+8+++_fixed.py when road_calibration.json is missing:
draw the centreline (upstream -> downstream), optional segment lengths, total length,
width references, then arbitrary reference lines. The result is saved as
calibrations/<name>.json and appears in the dashboard's "Road calibration" list.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

from app.core.config import settings
from app.pipeline.calibrator import RoadCalibrator
from app.pipeline.geometry import RoadMapper


def read_frame(video: Path, at_second: float):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video}")
    if at_second > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, at_second * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise OSError(f"Cannot read a frame from {video} at {at_second} s")
    return frame


def ask_total_length() -> float:
    """Same prompt and retry loop as 8+8+++_fixed.py."""
    while True:
        try:
            return float(input("Enter the total real length of the entire centreline (metres): "))
        except ValueError:
            print("Invalid input. Please enter a number.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a road calibration by clicking on a video frame (OpenCV windows)."
    )
    parser.add_argument("video", help="Video file name in VIDEOS_DIR, or a path to a video")
    parser.add_argument("name", help="Calibration name, saved as calibrations/<name>.json")
    parser.add_argument(
        "--at-second",
        type=float,
        default=0.0,
        help="Take the frame at this time instead of the first frame (default: 0, "
        "the first frame, as in 8+8+++_fixed.py)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing calibration")
    args = parser.parse_args(argv)

    if not args.name.replace("-", "").replace("_", "").isalnum():
        parser.error("name may only contain letters, digits, '-' and '_'")
    out = settings.calibrations_dir / f"{args.name}.json"
    if out.exists() and not args.force:
        parser.error(f"{out} already exists - choose another name or add --force")

    video = Path(args.video)
    if not video.is_file():
        video = settings.videos_dir / args.video
    if not video.is_file():
        parser.error(f"Video not found: {args.video} (also looked in {settings.videos_dir})")

    frame = read_frame(video, args.at_second)
    print(f"Calibrating on {video.name} at {args.at_second:g} s -> {out}")

    # ---- Same sequence as 8+8+++_fixed.py (lines 623-643) ----
    calibrator = RoadCalibrator(frame)
    calibrator.run_centreline()
    if len(calibrator.centreline) < 2:
        print("Calibration cancelled: the centreline needs at least 2 points.")
        return 1
    segment_lengths = calibrator.run_segment_lengths()
    total_len = ask_total_length()
    calibrator.run_width_references()
    print("\n--- Now add arbitrary reference lines (known lengths anywhere) ---")
    calibrator.run_arbitrary_references()
    calib_data = calibrator.get_calibration_data(total_len, segment_lengths=segment_lengths)
    cv2.destroyAllWindows()

    RoadMapper(calib_data)  # fail here, not later in a job, if the result is unusable
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(calib_data, f)
    print(f"Calibration saved: {out}")
    print(f"It now appears as '{args.name}' in the dashboard's Road calibration list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
