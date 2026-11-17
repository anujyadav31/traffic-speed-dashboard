"""Road geometry: image pixels -> road coordinates (distance along road, lateral offset).

Copied verbatim from `Vehicle detection and speed estimation/8+8+++_fixed.py`
(segment_intersection and RoadMapper) so speeds match the original script exactly.
Only this header was added.
"""

import math

import numpy as np


def segment_intersection(p1, p2, p3, p4):
    """Return (True, t) if segment p1-p2 crosses segment p3-p4, where t in [0,1]
    parametrizes the crossing point along p1-p2. Used to find where a clicked
    width-reference line actually crosses the centreline polyline."""
    r = (p2[0] - p1[0], p2[1] - p1[1])
    s = (p4[0] - p3[0], p4[1] - p3[1])
    rxs = r[0] * s[1] - r[1] * s[0]
    if abs(rxs) < 1e-9:
        return False, None
    qmp = (p3[0] - p1[0], p3[1] - p1[1])
    t = (qmp[0] * s[1] - qmp[1] * s[0]) / rxs
    u = (qmp[0] * r[1] - qmp[1] * r[0]) / rxs
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return True, t
    return False, None

# ============================================================


class RoadMapper:
    def __init__(self, calib_data):
        self.points = np.array(calib_data['centreline'], dtype=np.float32)
        self.cum_dist = np.array(calib_data['cumulative_dist'], dtype=np.float32)
        self.width_refs = calib_data.get('width_refs', [])
        self.arbitrary_refs = calib_data.get('arbitrary_refs', [])

        # Unified scale table: each entry (arc_len, m_per_px, side)
        self.scale_table = []
        self._build_scale_table()

        # Curvature calculation
        self.curvature_available = False
        self.curvature_s = []
        self.curvature_kappa = []
        self._compute_curvature()

    def _closest_point_on_polyline(self, pt):
        min_dist = float('inf')
        best_seg = 0
        best_t = 0.0
        for i in range(len(self.points) - 1):
            a = self.points[i]
            b = self.points[i+1]
            ab = b - a
            ap = np.array(pt) - a
            t = np.dot(ap, ab) / np.dot(ab, ab)
            t = max(0.0, min(1.0, t))
            proj = a + t * ab
            dist = np.linalg.norm(np.array(pt) - proj)
            if dist < min_dist:
                min_dist = dist
                best_seg = i
                best_t = t
        base_arc = self.cum_dist[best_seg]
        seg_length = self.cum_dist[best_seg+1] - self.cum_dist[best_seg]
        arc_len = base_arc + best_t * seg_length
        return best_seg, arc_len

    def _signed_lateral_px(self, pt):
        """Shared by image_to_world() and width-ref calibration so both sides of
        the road use the exact same sign convention for 'left' vs 'right'."""
        seg_idx, arc_len = self._closest_point_on_polyline(pt)
        a = self.points[seg_idx]
        b = self.points[seg_idx+1]
        ab = b - a
        v = np.array(pt) - a
        lateral_px = (ab[0]*v[1] - ab[1]*v[0]) / np.linalg.norm(ab)
        return arc_len, lateral_px

    def _build_scale_table(self):
        for ref in self.width_refs:
            if 'left_pt' in ref and 'right_pt' in ref:
                # Determine each edge's side from its own signed lateral offset
                # instead of assuming "left click" == -1 side, since the sign
                # produced by the cross-product formula depends on centreline
                # direction and isn't guaranteed to match click order.
                left_m = ref['left_m']
                right_m = ref['right_m']
                arc_len_l, lat_l = self._signed_lateral_px(ref['left_pt'])
                arc_len_r, lat_r = self._signed_lateral_px(ref['right_pt'])
                if abs(lat_l) > 0:
                    side_l = 1 if lat_l >= 0 else -1
                    self.scale_table.append((arc_len_l, left_m / abs(lat_l), side_l))
                if abs(lat_r) > 0:
                    side_r = 1 if lat_r >= 0 else -1
                    self.scale_table.append((arc_len_r, right_m / abs(lat_r), side_r))
            elif 'left_px' in ref and 'right_px' in ref:
                # Backward compatibility with calibration files saved before
                # left_pt/right_pt were recorded (falls back to assumed sides).
                pt = ref['pt_centre']
                _, arc_len = self._closest_point_on_polyline(pt)
                left_px = ref['left_px']
                right_px = ref['right_px']
                left_m = ref['left_m']
                right_m = ref['right_m']
                if left_px > 0:
                    self.scale_table.append((arc_len, left_m / left_px, -1))
                if right_px > 0:
                    self.scale_table.append((arc_len, right_m / right_px, +1))
            else:
                pt = ref['pt_centre']
                _, arc_len = self._closest_point_on_polyline(pt)
                total_px = ref['width_px']
                total_m = ref['real_width_m']
                if total_px > 0:
                    m_per_px = total_m / total_px
                    self.scale_table.append((arc_len, m_per_px, -1))
                    self.scale_table.append((arc_len, m_per_px, +1))
        for ref in self.arbitrary_refs:
            pt1 = ref['pt1']
            pt2 = ref['pt2']
            mid_pt = ((pt1[0]+pt2[0])/2, (pt1[1]+pt2[1])/2)
            _, arc_len = self._closest_point_on_polyline(mid_pt)
            px_len = math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])
            if px_len > 0:
                m_per_px = ref['real_length_m'] / px_len
                self.scale_table.append((arc_len, m_per_px, -1))
                self.scale_table.append((arc_len, m_per_px, +1))
        self.scale_table.sort(key=lambda x: x[0])

    def _compute_curvature(self):
        if len(self.points) < 3:
            return
        n = len(self.points)
        for i in range(1, n-1):
            p_prev = self.points[i-1]
            p_curr = self.points[i]
            p_next = self.points[i+1]
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            if len1 == 0 or len2 == 0:
                continue
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            dtheta = math.atan2(cross, dot)
            # ds must be in real-world metres (via cum_dist), not pixels, so that
            # kappa = dtheta/ds has units rad/metre. physical_distance() multiplies
            # kappa by a lateral offset that's already in metres (kappa * l must be
            # dimensionless) - using pixel-space ds here made that correction wrong
            # by whatever the local px-per-metre scale happened to be.
            ds = (self.cum_dist[i+1] - self.cum_dist[i-1]) / 2.0
            if ds <= 0:
                continue
            kappa = dtheta / ds
            s = self.cum_dist[i]
            self.curvature_s.append(s)
            self.curvature_kappa.append(kappa)
        if self.curvature_s:
            self.curvature_available = True
            if self.curvature_s[0] > self.cum_dist[0]:
                self.curvature_s.insert(0, self.cum_dist[0])
                self.curvature_kappa.insert(0, self.curvature_kappa[0])
            if self.curvature_s[-1] < self.cum_dist[-1]:
                self.curvature_s.append(self.cum_dist[-1])
                self.curvature_kappa.append(self.curvature_kappa[-1])

    def image_to_world(self, pt):
        arc_len, lateral_px = self._signed_lateral_px(pt)
        m_per_px = self._get_scale_idw(arc_len, lateral_px)
        lateral_m = lateral_px * m_per_px
        return arc_len, lateral_m

    def _get_scale_idw(self, arc_len, lateral_px, power=2.0):
        """Interpolate scale using inverse-distance weighting (IDW)."""
        if not self.scale_table:
            return 1.0
        side = 1 if lateral_px >= 0 else -1
        entries = [(s, m) for (s, m, sd) in self.scale_table if sd == side]
        if not entries:
            entries = [(s, m) for (s, m, sd) in self.scale_table]
        if not entries:
            return 1.0

        # If only one entry, return its scale
        if len(entries) == 1:
            return entries[0][1]

        # If arc_len is outside range, use nearest
        arcs = np.array([e[0] for e in entries])
        if arc_len <= arcs.min():
            return entries[np.argmin(arcs)][1]
        if arc_len >= arcs.max():
            return entries[np.argmax(arcs)][1]

        # Compute IDW weights for all entries
        distances = np.abs(arcs - arc_len)
        # Avoid division by zero
        min_dist = distances.min()
        if min_dist < 1e-9:
            # Exact match, return that scale
            idx = np.argmin(distances)
            return entries[idx][1]

        weights = 1.0 / (distances ** power)
        weights /= weights.sum()
        scale = sum(w * m for w, (_, m) in zip(weights, entries))
        return scale

    def _curvature_at(self, s):
        if not self.curvature_available:
            return 0.0
        if s <= self.curvature_s[0]:
            return self.curvature_kappa[0]
        if s >= self.curvature_s[-1]:
            return self.curvature_kappa[-1]
        for i in range(len(self.curvature_s)-1):
            if self.curvature_s[i] <= s <= self.curvature_s[i+1]:
                t = (s - self.curvature_s[i]) / (self.curvature_s[i+1] - self.curvature_s[i])
                return self.curvature_kappa[i] + t * (self.curvature_kappa[i+1] - self.curvature_kappa[i])
        return 0.0

    def physical_distance(self, world1, world2):
        s1, l1 = world1
        s2, l2 = world2
        ds = s2 - s1
        dl = l2 - l1
        if self.curvature_available:
            s_mid = (s1 + s2) / 2.0
            l_avg = (l1 + l2) / 2.0
            kappa = self._curvature_at(s_mid)
            # Exact parallel curve arc-length differential: ds_parallel = (1 - kappa*l) * ds
            adjusted_ds = (1.0 - kappa * l_avg) * ds
            return math.hypot(adjusted_ds, dl)
        else:
            return math.hypot(ds, dl)
