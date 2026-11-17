"""Interactive road calibration (OpenCV windows + terminal input).

The RoadCalibrator class below is copied verbatim from
`Vehicle detection and speed estimation/8+8+++_fixed.py` (lines 29-332), so new
calibrations are made with exactly the same tool and maths as the original script.
Only this header and the imports were added; segment_intersection comes from
geometry.py, where it is also a verbatim copy of the same function.
"""
# Keep the copied code byte-for-byte: no lint fixes, no reformatting.
# ruff: noqa
# fmt: off

import math

import cv2
import numpy as np

from app.pipeline.geometry import segment_intersection

# 1. CALIBRATION MODULE (with direction arrow & error handling)
# ============================================================
class RoadCalibrator:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.centreline = []               # list of (x,y) image points
        self.width_refs = []               # list of dicts
        self.arbitrary_refs = []           # list of dicts
        self.mode = 'centreline'
        self.width_start = None
        self.arb_start = None
        self.mouse_pos = None              # for live drawing of width line

    def _line_intersect_centreline(self, p1, p2):
        for i in range(len(self.centreline) - 1):
            a = self.centreline[i]
            b = self.centreline[i+1]
            hit, t = segment_intersection(p1, p2, a, b)
            if hit:
                return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
        return None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == 'centreline':
                self.centreline.append((x, y))
                print(f"Centreline point added: ({x},{y})")
            elif self.mode == 'width':
                if self.width_start is None:
                    self.width_start = (x, y)
                    print("Left edge point set. Now click the opposite (right) edge.")
                else:
                    left_pt = self.width_start
                    right_pt = (x, y)
                    centre = self._line_intersect_centreline(left_pt, right_pt)
                    if centre is None:
                        # Width line didn't cross the centreline (e.g. imprecise
                        # click); fall back to the straight midpoint instead.
                        centre = ((left_pt[0] + right_pt[0]) / 2.0, (left_pt[1] + right_pt[1]) / 2.0)
                        print("Warning: width line did not intersect the centreline; using midpoint instead.")
                    cx, cy = centre

                    left_px = math.hypot(left_pt[0] - cx, left_pt[1] - cy)
                    right_px = math.hypot(right_pt[0] - cx, right_pt[1] - cy)
                    total_px = left_px + right_px

                    # Input with error handling
                    while True:
                        try:
                            left_m = float(input("Enter real width from centreline to LEFT edge (metres): "))
                            break
                        except ValueError:
                            print("Invalid input. Please enter a number.")
                    while True:
                        try:
                            right_m = float(input("Enter real width from centreline to RIGHT edge (metres): "))
                            break
                        except ValueError:
                            print("Invalid input. Please enter a number.")

                    self.width_refs.append({
                        'pt_centre': centre,
                        'left_pt': left_pt,
                        'right_pt': right_pt,
                        'left_px': left_px,
                        'right_px': right_px,
                        'left_m': left_m,
                        'right_m': right_m,
                        'width_px': total_px,
                        'real_width_m': left_m + right_m
                    })
                    print(f"Width reference added: centre=({cx:.1f},{cy:.1f}), "
                          f"left={left_px:.1f}px ({left_m}m), right={right_px:.1f}px ({right_m}m)")
                    self.width_start = None
            elif self.mode == 'arbitrary':
                if self.arb_start is None:
                    self.arb_start = (x, y)
                    print("Arbitrary reference start point set. Now click the end point.")
                else:
                    dx = x - self.arb_start[0]
                    dy = y - self.arb_start[1]
                    px_length = math.hypot(dx, dy)
                    while True:
                        try:
                            real_len = float(input("Enter the real length of this line (metres): "))
                            break
                        except ValueError:
                            print("Invalid input. Please enter a number.")
                    self.arbitrary_refs.append({
                        'pt1': self.arb_start,
                        'pt2': (x, y),
                        'real_length_m': real_len
                    })
                    print(f"Arbitrary reference added: length px={px_length:.1f}, real={real_len}m")
                    self.arb_start = None

    def run_centreline(self):
        self.mode = 'centreline'
        cv2.imshow("Calibration - Draw Centreline", self.frame)
        cv2.setMouseCallback("Calibration - Draw Centreline", self.mouse_callback)
        print("Click points along the road centreline. Press 'd' when finished.")
        print("Press 'u' to undo the last point.")
        print("NOTE: Draw the centreline in the direction of traffic flow (from upstream to downstream).")
        while True:
            disp = self.frame.copy()
            if len(self.centreline) > 1:
                pts = np.array(self.centreline, np.int32)
                cv2.polylines(disp, [pts], False, (0,255,0), 2)
                # Draw an arrow at the last segment to show direction
                if len(self.centreline) >= 2:
                    p1 = self.centreline[-2]
                    p2 = self.centreline[-1]
                    # compute angle
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    angle = math.atan2(dy, dx)
                    # arrow tip length
                    tip_len = 20
                    tip_angle = np.pi/6
                    x_tip = p2[0]
                    y_tip = p2[1]
                    # draw arrow using cv2.arrowedLine
                    cv2.arrowedLine(disp, p1, p2, (0,255,255), 2, tipLength=0.3)
            for pt in self.centreline:
                cv2.circle(disp, pt, 4, (0,0,255), -1)
            cv2.imshow("Calibration - Draw Centreline", disp)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('d'):
                break
            if key == ord('u'):
                if self.centreline:
                    removed = self.centreline.pop()
                    print(f"Undo: removed point {removed}")
                else:
                    print("No points to undo.")
        cv2.destroyWindow("Calibration - Draw Centreline")

    def run_segment_lengths(self):
        n_seg = len(self.centreline) - 1
        segment_lengths = []
        print("\n--- Now enter real lengths for individual centreline segments (optional) ---")
        for i in range(n_seg):
            pt1 = self.centreline[i]
            pt2 = self.centreline[i+1]
            disp = self.frame.copy()
            if len(self.centreline) > 1:
                pts = np.array(self.centreline, np.int32)
                cv2.polylines(disp, [pts], False, (0,255,0), 2)
            cv2.line(disp, pt1, pt2, (0,255,255), 3)
            cv2.circle(disp, pt1, 6, (0,255,255), -1)
            cv2.circle(disp, pt2, 6, (0,255,255), -1)
            cv2.putText(disp, f"Seg {i}", (pt1[0]+10, pt1[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.imshow("Calibration - Segment Length", disp)
            cv2.waitKey(1)

            inp = input(f"  Segment {i}: real length in metres (press Enter to skip): ")
            if inp.strip() == "":
                segment_lengths.append(None)
                print("    skipped.")
            else:
                try:
                    val = float(inp)
                    segment_lengths.append(val)
                    print(f"    set to {val} m")
                except ValueError:
                    print("    Invalid input, skipping.")
                    segment_lengths.append(None)
        cv2.destroyWindow("Calibration - Segment Length")
        return segment_lengths

    def run_width_references(self):
        self.mode = 'width'
        cv2.imshow("Calibration - Width References", self.frame)
        cv2.setMouseCallback("Calibration - Width References", self.mouse_callback)
        print("Add width references: click left edge, then right edge, then enter left and right widths.")
        print("Press 'n' to start a new reference, 'q' when finished.")
        print("Press 'u' to undo the last click (cancels current point or removes last reference).")
        while True:
            disp = self.frame.copy()
            if len(self.centreline) > 1:
                cv2.polylines(disp, [np.array(self.centreline, np.int32)], False, (0,255,0), 2)
            for ref in self.width_refs:
                if 'left_pt' in ref and 'right_pt' in ref:
                    lp = (int(ref['left_pt'][0]), int(ref['left_pt'][1]))
                    rp = (int(ref['right_pt'][0]), int(ref['right_pt'][1]))
                    cv2.line(disp, lp, rp, (255,0,0), 2)
                cv2.circle(disp, (int(ref['pt_centre'][0]), int(ref['pt_centre'][1])), 5, (255,0,0), -1)
            if self.width_start is not None and self.mouse_pos is not None:
                cv2.line(disp, self.width_start, self.mouse_pos, (0,255,255), 2)
                cv2.circle(disp, self.width_start, 6, (0,255,0), -1)
            cv2.imshow("Calibration - Width References", disp)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break
            if key == ord('n'):
                self.width_start = None
                print("Ready for next width reference.")
            if key == ord('u'):
                if self.width_start is not None:
                    self.width_start = None
                    print("Undo: cancelled current first point.")
                elif self.width_refs:
                    removed = self.width_refs.pop()
                    print(f"Undo: removed last width reference at centre {removed['pt_centre']}")
                else:
                    print("Nothing to undo.")
        cv2.destroyWindow("Calibration - Width References")

    def run_arbitrary_references(self):
        self.mode = 'arbitrary'
        cv2.imshow("Calibration - Arbitrary References", self.frame)
        cv2.setMouseCallback("Calibration - Arbitrary References", self.mouse_callback)
        print("Add arbitrary reference lines: click start point, then end point, then enter real length.")
        print("Press 'n' to start a new line, 'q' when finished.")
        print("Press 'u' to undo the last click (cancels current point or removes last reference).")
        while True:
            disp = self.frame.copy()
            if len(self.centreline) > 1:
                cv2.polylines(disp, [np.array(self.centreline, np.int32)], False, (0,255,0), 2)
            for ref in self.arbitrary_refs:
                cv2.line(disp, ref['pt1'], ref['pt2'], (255,0,255), 2)
                mid = ((ref['pt1'][0]+ref['pt2'][0])//2, (ref['pt1'][1]+ref['pt2'][1])//2)
                cv2.circle(disp, mid, 5, (255,0,255), -1)
            if self.arb_start is not None and self.mouse_pos is not None:
                cv2.line(disp, self.arb_start, self.mouse_pos, (0,255,255), 2)
                cv2.circle(disp, self.arb_start, 6, (0,255,0), -1)
            cv2.imshow("Calibration - Arbitrary References", disp)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break
            if key == ord('n'):
                self.arb_start = None
                print("Ready for next arbitrary reference line.")
            if key == ord('u'):
                if self.arb_start is not None:
                    self.arb_start = None
                    print("Undo: cancelled current start point.")
                elif self.arbitrary_refs:
                    removed = self.arbitrary_refs.pop()
                    print(f"Undo: removed last arbitrary reference between {removed['pt1']} and {removed['pt2']}")
                else:
                    print("Nothing to undo.")
        cv2.destroyWindow("Calibration - Arbitrary References")

    def get_calibration_data(self, total_real_length, segment_lengths=None):
        pts = np.array(self.centreline)
        seg_len_px = []
        for i in range(1, len(pts)):
            seg_len_px.append(np.linalg.norm(pts[i] - pts[i-1]))

        if segment_lengths is not None and any(l is not None for l in segment_lengths):
            specified = [(i, l) for i, l in enumerate(segment_lengths) if l is not None]
            sum_specified = sum(l for _, l in specified)
            if sum_specified > total_real_length:
                print("Warning: sum of specified segment lengths exceeds total length – clipping.")
                factor = total_real_length / sum_specified
                sum_specified = total_real_length
                for idx, val in specified:
                    segment_lengths[idx] = val * factor

            unspecified_indices = [i for i, l in enumerate(segment_lengths) if l is None]
            total_px_unspecified = sum(seg_len_px[i] for i in unspecified_indices)
            remaining = total_real_length - sum_specified
            if remaining < 0:
                remaining = 0
            if total_px_unspecified > 0 and remaining > 0:
                px_to_m = remaining / total_px_unspecified
            else:
                px_to_m = 0.0

            final_lengths = []
            for i in range(len(seg_len_px)):
                if segment_lengths[i] is not None:
                    final_lengths.append(segment_lengths[i])
                else:
                    final_lengths.append(seg_len_px[i] * px_to_m)

            current_total = sum(final_lengths)
            if current_total > 0:
                correction = total_real_length / current_total
                final_lengths = [l * correction for l in final_lengths]

            cum_dist = [0.0]
            for l in final_lengths:
                cum_dist.append(cum_dist[-1] + l)
        else:
            total_px = sum(seg_len_px)
            scale = total_real_length / total_px if total_px > 0 else 1.0
            cum_dist = [0.0]
            for s in seg_len_px:
                cum_dist.append(cum_dist[-1] + s * scale)

        return {
            'centreline': self.centreline,
            'cumulative_dist': cum_dist,
            'width_refs': self.width_refs,
            'arbitrary_refs': self.arbitrary_refs,
            'segment_lengths': segment_lengths,
            'total_real_length': total_real_length
        }
