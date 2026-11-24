"""Shared helpers for the fine-tuning tools (mcap_estimate_offset.py, build_finetune_dataset.py).

Stage-1 run directory (tools/mcap_semantic_mapping.py):
  meta.json      origin = [x, y, z] of the first pose in the pose source's world frame (UTM for NovAtel odometry)
  frames.jsonl   one keyframe per line: idx, stamp_ns, map (png path), pose = [x, y, yaw] relative to origin
  maps/*.png     300 x 150 ego-centric semantic maps: row 0 = 30 m ahead, column 0 = 15 m to the left, 0.2 m/px
"""
import json
from pathlib import Path

import cv2
import numpy as np

MAP_RES, MAP_X, MAP_Y = 0.2, 30.0, 15.0  # m/px, half-length forward, half-width lateral
# semantic map classes as stored in the PNGs (BGR of semantic_mapping/src/node_config/base_cfg.py LABEL_COLORS);
# the renderer blends neighbouring classes, so pixels are matched with a tolerance
CLASS_BGR = dict(road=(128, 64, 128), crosswalk=(200, 140, 140), lane=(255, 255, 255), sidewalk=(232, 35, 244))


def load_run(run):
    run = Path(run)
    meta = json.loads((run / 'meta.json').read_text())
    frames = [json.loads(l) for l in (run / 'frames.jsonl').read_text().splitlines() if l.strip()]
    if meta.get('origin') is None:
        raise SystemExit(f'{run}/meta.json has no pose origin; rerun stage 1')
    return meta, frames


def class_masks(sem_bgr, tol=30):
    img = sem_bgr.astype(np.int16)
    return {c: np.abs(img - np.array(bgr, np.int16)).sum(2) <= tol for c, bgr in CLASS_BGR.items()}


def curb_mask(masks):
    """Road cells touching sidewalk cells: where the road edge (curb) was actually observed."""
    side = cv2.dilate(masks['sidewalk'].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return masks['road'] & side


def paint_mask(masks, max_width_m=0.8):
    """Lane pixels that can be paint: HRNet also labels large bright areas (snow banks, sunlit concrete) as lane
    marking, and those blobs - far wider than any painted line - would swamp the real lines in an alignment fit."""
    lane = masks['lane'].astype(np.uint8)
    k = int(round(max_width_m / MAP_RES)) | 1
    blobs = cv2.morphologyEx(lane, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    blobs = cv2.dilate(blobs, np.ones((5, 5), np.uint8))
    return masks['lane'] & (blobs == 0)


def ego_pose_world(frame, origin):
    x, y, yaw = frame['pose']
    return np.array([origin[0] + x, origin[1] + y]), yaw


def pixels_to_world(mask, frame, origin, stride=1):
    """World (x, y) of the set pixels of an ego-centric semantic map (same pixel convention as stage 2)."""
    v, u = np.nonzero(mask[::stride, ::stride])
    v, u = v * stride, u * stride
    ex, ey = MAP_X - v * MAP_RES, MAP_Y - u * MAP_RES  # ego frame: x forward, y left
    (wx, wy), yaw = ego_pose_world(frame, origin)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([wx + c * ex - s * ey, wy + s * ex + c * ey], 1)


def dedupe(pts, cell=0.2):
    """One point per cell: keyframe maps overlap heavily, so the same cell is seen by many keyframes."""
    if len(pts) == 0:
        return pts
    _, idx = np.unique(np.floor(pts / cell).astype(np.int64), axis=0, return_index=True)
    return pts[np.sort(idx)]


class WorldRaster:
    """A metric grid over part of the world, for distance fields of map elements (row = y, column = x)."""

    def __init__(self, lo, hi, res=0.2):
        self.lo, self.res = np.asarray(lo, float), res
        self.w = int(np.ceil((hi[0] - lo[0]) / res)) + 1
        self.h = int(np.ceil((hi[1] - lo[1]) / res)) + 1

    def px(self, xy):
        return (np.asarray(xy, float) - self.lo) / self.res

    def blank(self):
        return np.zeros((self.h, self.w), np.uint8)

    def draw_lines(self, lines, mask=None, thickness=1):
        mask = self.blank() if mask is None else mask
        pl = [np.round(self.px(l)).astype(np.int32) for l in lines if len(l) > 1]
        if pl:
            cv2.polylines(mask, pl, False, 1, thickness)
        return mask

    def fill(self, geom, mask=None):
        """Rasterise a shapely (Multi)Polygon, holes included."""
        mask = self.blank() if mask is None else mask
        for poly in getattr(geom, 'geoms', [geom]):
            if poly.is_empty or poly.geom_type != 'Polygon':
                continue
            cv2.fillPoly(mask, [np.round(self.px(np.array(poly.exterior.coords)[:, :2])).astype(np.int32)], 1)
            holes = [np.round(self.px(np.array(r.coords)[:, :2])).astype(np.int32) for r in poly.interiors]
            if holes:
                cv2.fillPoly(mask, holes, 0)
        return mask

    def distance(self, mask):
        """Distance (m) from every cell to the nearest set cell of mask."""
        if not mask.any():
            return np.full(mask.shape, 1e3, np.float32)
        return cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5) * self.res

    def sample(self, field, xy, outside=1e3):
        """Bilinear lookup of a field at world points (smooth in the offset, unlike nearest-cell lookup)."""
        p = self.px(xy)
        x, y = p[:, 0], p[:, 1]
        x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        ok = (x0 >= 0) & (y0 >= 0) & (x0 < self.w - 1) & (y0 < self.h - 1)
        out = np.full(len(p), outside, np.float32)
        if ok.any():
            x0, y0, fx, fy = x0[ok], y0[ok], (x - np.floor(x))[ok], (y - np.floor(y))[ok]
            f = field
            out[ok] = (f[y0, x0] * (1 - fx) * (1 - fy) + f[y0, x0 + 1] * fx * (1 - fy) +
                       f[y0 + 1, x0] * (1 - fx) * fy + f[y0 + 1, x0 + 1] * fx * fy)
        return out
