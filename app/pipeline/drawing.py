"""Drawing helpers for the annotated video.

Copied verbatim from `Vehicle detection and speed estimation/8+8+++_fixed.py`
(trajectory settings, compute_iou, generate_random_color, trail smoothing,
draw_trajectory, most_frequent_class). Only this header and the imports were added.
"""

import math

import cv2
import numpy as np

# ---- Trajectory display (press 't' during playback to toggle) ----
SHOW_TRAJECTORIES = True
TRAJECTORY_SECONDS = 5.0                  # how much past motion to draw as a trail
TRAJECTORY_MAX_POINTS = 300               # hard cap on stored trail points per track
TRAJECTORY_MIN_DRAW_PX = 1.0              # skip near-duplicate points (parked vehicles)
# Smoothing of the drawn trail only. None of this touches track_history, so the
# speed calculation and the CSV still use the raw centroids exactly as before.
TRAJECTORY_RESAMPLE_PX = 6.0              # uniform arc-length spacing before filtering
TRAJECTORY_SMOOTH_SIGMA = 6.0#4.0             # Gaussian low-pass width, in resampled steps
TRAJECTORY_SMOOTH_PASSES = 1              # Chaikin corner-cutting passes (0 = off)
TRAJECTORY_FADE_BANDS = 12                # colour steps along the trail
TRAJECTORY_THICKNESS = 1                  # uniform line width for the whole trail

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0

def generate_random_color():
    import random
    hue = random.randint(0, 179)
    sat = random.randint(200, 255)
    val = random.randint(200, 255)
    hsv = np.uint8([[[hue, sat, val]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

def resample_polyline(pts, step):
    """Re-space the trail at uniform arc length. Detections arrive per frame, so
    a near vehicle leaves points far apart and a distant one leaves them almost
    on top of each other; without this the filter below would smooth the two by
    wildly different amounts."""
    if len(pts) < 2 or step <= 0:
        return [(float(x), float(y)) for x, y in pts]
    out = [(float(pts[0][0]), float(pts[0][1]))]
    residual = 0.0
    for i in range(1, len(pts)):
        x0, y0 = pts[i-1]
        x1, y1 = pts[i]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 1e-9:
            continue
        d = step - residual
        while d <= seg:
            t = d / seg
            out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
            d += step
        residual = seg - (d - step)
    lx, ly = float(pts[-1][0]), float(pts[-1][1])
    if math.hypot(out[-1][0] - lx, out[-1][1] - ly) > 1e-6:
        out.append((lx, ly))
    return out

def gaussian_smooth(pts, sigma):
    """Symmetric Gaussian low-pass along the trail. A boxcar average of the same
    length leaves ripple behind; a Gaussian actually attenuates it, which is the
    difference between a trail that looks 'less jagged' and one that looks smooth.
    Both endpoints are pinned so the trail still starts and ends on real
    positions - in particular the head stays exactly on the vehicle."""
    n = len(pts)
    if n < 3 or sigma <= 0:
        return [(float(x), float(y)) for x, y in pts]
    radius = max(1, int(round(3 * sigma)))
    k = np.exp(-(np.arange(-radius, radius + 1) ** 2) / (2.0 * sigma * sigma))
    k /= k.sum()
    arr = np.asarray(pts, dtype=np.float64)
    pad = np.pad(arr, ((radius, radius), (0, 0)), mode='edge')
    out = np.empty_like(arr)
    out[:, 0] = np.convolve(pad[:, 0], k, mode='valid')
    out[:, 1] = np.convolve(pad[:, 1], k, mode='valid')
    out[0], out[-1] = arr[0], arr[-1]
    return [(float(x), float(y)) for x, y in out]

def chaikin(pts, passes):
    """Chaikin corner cutting: each pass replaces every segment by points at 1/4
    and 3/4 along it, rounding the corners without overshooting them. Endpoints
    are kept so the trail stays anchored to the vehicle."""
    for _ in range(max(0, passes)):
        if len(pts) < 3:
            break
        out = [pts[0]]
        for i in range(len(pts) - 1):
            (px, py), (qx, qy) = pts[i], pts[i+1]
            out.append((0.75*px + 0.25*qx, 0.75*py + 0.25*qy))
            out.append((0.25*px + 0.75*qx, 0.25*py + 0.75*qy))
        out.append(pts[-1])
        pts = out
    return pts

def draw_trajectory(img, pts, color):
    """Draw a track's recent path as a trail of uniform thickness that fades
    towards the oldest point. Points are smoothed and drawn with 1/16-pixel
    precision, in a few constant-shade bands rather than one call per segment,
    so the curve is smooth without the per-segment cost."""
    if len(pts) < 2:
        return
    pts = resample_polyline(pts, TRAJECTORY_RESAMPLE_PX)
    pts = gaussian_smooth(pts, TRAJECTORY_SMOOTH_SIGMA)
    pts = chaikin(pts, TRAJECTORY_SMOOTH_PASSES)
    n = len(pts)
    if n < 2:
        return
    shift = 4                                    # sub-pixel: coords are in 1/16 px
    scale = 1 << shift
    arr = np.array([(round(x * scale), round(y * scale)) for x, y in pts],
                   dtype=np.int32)
    bands = max(1, min(TRAJECTORY_FADE_BANDS, n - 1))
    for b in range(bands):
        lo = int(b * (n - 1) / bands)
        hi = int((b + 1) * (n - 1) / bands) + 1
        if hi - lo < 2:
            continue
        frac = (b + 0.5) / bands                 # 0 = oldest, 1 = newest
        fade = 0.25 + 0.75 * frac
        faded = (int(color[0]*fade), int(color[1]*fade), int(color[2]*fade))
        cv2.polylines(img, [arr[lo:hi]], False, faded, TRAJECTORY_THICKNESS,
                      cv2.LINE_AA, shift)
    cv2.circle(img, (int(arr[-1][0]), int(arr[-1][1])), 3 * scale, color, -1,
               cv2.LINE_AA, shift)

def most_frequent_class(history):
    if not history:
        return -1
    from collections import Counter
    counter = Counter(history)
    return counter.most_common(1)[0][0]
