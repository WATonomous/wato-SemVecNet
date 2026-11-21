#!/usr/bin/env python3
"""Stage 3 of the MCAP pipeline: score SemVecNet's predictions against a Lanelet2 HD map (e.g. ring_road.osm).

For every keyframe the ground truth is cut out of the Lanelet2 map exactly the way MapTRv2 builds its nuScenes GT
(SemVecNet/tools/maptrv2/custom_nusc_map_converter.py), in the frame the model predicts in: a 30 x 60 m patch
around a virtual nuScenes LIDAR_TOP 0.94 m ahead of the ego, x = right, y = forward.
  divider       lane bounds shared by two lanelets (lane / road dividers), minus the parts inside junctions
  ped_crossing  outline of the union of crosswalk lanelets          (ped_poly_geoms_to_instances)
  boundary      outline of the union of all road lanelets           (poly_geoms_to_instances)
  centerline    lanelet centrelines chained through successors      (union_centerline)
Scores are MapTR's own: chamfer-distance AP at 0.5 / 1.0 / 1.5 m via map_utils.mean_ap (format_res_gt_by_classes +
eval_map), plus precision / recall of the predictions above --score-thr.

The map only covers the mapped roads, so by default predictions that lie mostly outside the mapped area
(lanelets + --map-buffer) are not counted ("mapped" scores); "full" scores count everything in the patch.
Frames where the ego is more than --max-offmap metres from any road lanelet are skipped.

The ego pose comes from stage 1 (meta.json origin + frames.jsonl pose). It must be in the same projected CRS as the
map: NovAtel odometry (/novatel/oem7/odom) is UTM, and the map's lat/lon are projected to the UTM zone of the map.
A constant pose/map offset is estimated from the confident divider/boundary predictions and reported; scores are
also given with that offset removed, to separate localization error from perception error.
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from shapely import affinity, ops
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, box

MAP_CLASSES = ['divider', 'ped_crossing', 'boundary', 'centerline']
CLASS_BGR = [(0, 165, 255), (255, 0, 0), (0, 0, 255), (0, 200, 0)]  # orange, blue, red, green
LIDAR_X_IN_EGO = 0.943713          # same virtual LiDAR offset as tools/mcap_infer_semvecnet.py
PC_RANGE = [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0]
PATCH_W, PATCH_H = PC_RANGE[3] - PC_RANGE[0], PC_RANGE[4] - PC_RANGE[1]  # 30 (x, lateral), 60 (y, forward)
CHAMFER_THRESHOLDS = [0.5, 1.0, 1.5]
MAP_RES, MAP_X, MAP_Y = 0.2, 30.0, 15.0


# --------------------------------------------------------------------------- Lanelet2 map


def load_lanelet2(path, epsg=None):
    """Parse a Lanelet2 .osm into road / crosswalk lanelets with bounds in a projected CRS (UTM by default)."""
    from pyproj import Transformer
    root = ET.parse(path).getroot()
    ids, lonlat = [], []
    for n in root.iter('node'):
        ids.append(n.get('id'))
        lonlat.append((float(n.get('lon')), float(n.get('lat'))))
    lonlat = np.array(lonlat)
    if epsg is None:
        lon0, lat0 = lonlat.mean(0)
        epsg = (32600 if lat0 >= 0 else 32700) + int((lon0 + 180) // 6) + 1
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lonlat[:, 0], lonlat[:, 1])
    nodes = dict(zip(ids, np.stack([x, y], 1)))
    ways = {w.get('id'): [nd.get('ref') for nd in w.iter('nd')] for w in root.iter('way')}

    lanelets = []
    for rel in root.iter('relation'):
        tags = {t.get('k'): t.get('v') for t in rel.iter('tag')}
        if tags.get('type') != 'lanelet':
            continue
        mem = {m.get('role'): m.get('ref') for m in rel.iter('member') if m.get('type') == 'way'}
        if 'left' not in mem or 'right' not in mem or mem['left'] not in ways or mem['right'] not in ways:
            continue
        L = np.array([nodes[n] for n in ways[mem['left']]])
        R = np.array([nodes[n] for n in ways[mem['right']]])
        # Lanelet2 keeps a bound shared by opposite lanes in one direction; orient left like right
        if np.linalg.norm(L[0] - R[-1]) + np.linalg.norm(L[-1] - R[0]) < np.linalg.norm(L[0] - R[0]) + np.linalg.norm(L[-1] - R[-1]):
            L = L[::-1]
        lanelets.append(dict(id=rel.get('id'), subtype=tags.get('subtype', 'road'), name=tags.get('name', ''),
                             left_way=mem['left'], right_way=mem['right'], left=L, right=R))
    return lanelets, epsg


def resample(line, step=0.5):
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(np.ceil(s[-1] / step)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, line[:, 0]), np.interp(t, s, line[:, 1])], 1)


class WorldGT:
    """The map's vector elements in world coordinates, ready to be cut into per-frame patches."""

    def __init__(self, lanelets, map_buffer):
        road = [ll for ll in lanelets if ll['subtype'] != 'crosswalk']
        cross = [ll for ll in lanelets if ll['subtype'] == 'crosswalk']
        for ll in road + cross:
            ll['poly'] = Polygon(np.vstack([ll['left'], ll['right'][::-1]])).buffer(0)

        # successors: A's end (left, right) == B's start
        key = lambda p: (round(float(p[0]), 1), round(float(p[1]), 1))
        starts = {}
        for ll in road:
            starts.setdefault((key(ll['left'][0]), key(ll['right'][0])), []).append(ll['id'])
        for ll in road:
            ll['succ'] = [s for s in starts.get((key(ll['left'][-1]), key(ll['right'][-1])), []) if s != ll['id']]
        for ll in road:
            ll['pred'] = [o['id'] for o in road if ll['id'] in o['succ']]
            n = max(len(resample(ll['left'])), len(resample(ll['right'])))
            t = np.linspace(0, 1, n)
            center = (self._at(ll['left'], t) + self._at(ll['right'], t)) / 2
            ll['center'] = LineString(center)

        # junctions: road lanelets that overlap without being neighbours or successors
        bound_use = {}
        for ll in road:
            for w in (ll['left_way'], ll['right_way']):
                bound_use.setdefault(w, []).append(ll)
        overlaps = []
        for i, a in enumerate(road):
            for b in road[i + 1:]:
                if {a['left_way'], a['right_way']} & {b['left_way'], b['right_way']} or \
                        b['id'] in a['succ'] or a['id'] in b['succ'] or not a['poly'].intersects(b['poly']):
                    continue
                inter = a['poly'].intersection(b['poly'])
                if inter.area > 1.0:
                    overlaps.append(inter)
        self.junctions = ops.unary_union(overlaps).buffer(1.0) if overlaps else Polygon()

        self.dividers = []
        for w, lls in bound_use.items():
            if len(lls) < 2:
                continue
            ll = lls[0]
            coords = ll['left'] if ll['left_way'] == w else ll['right']
            line = LineString(coords).difference(self.junctions)
            self.dividers += [g for g in getattr(line, 'geoms', [line]) if not g.is_empty and g.length > 0.5]
        # where a lanelet ends without a successor (or starts without a predecessor) the map simply stops: the
        # road outline closes there with a line across the lane that does not exist on the road - not GT
        caps = [LineString([ll['left'][-1], ll['right'][-1]]) for ll in road if not ll['succ']] + \
               [LineString([ll['left'][0], ll['right'][0]]) for ll in road if not ll['pred']]
        self.map_ends = ops.unary_union([c.buffer(0.5, cap_style=2) for c in caps]) if caps else Polygon()
        self.roads = [ll['poly'] for ll in road]
        self.road_union = ops.unary_union([p.buffer(0.05) for p in self.roads]).buffer(-0.05)
        self.crosswalks = [ll['poly'] for ll in cross]
        self.lanelets = {ll['id']: ll for ll in road}
        self.mapped = ops.unary_union(self.roads + self.crosswalks).buffer(map_buffer)
        self.n_road, self.n_cross = len(road), len(cross)

    @staticmethod
    def _at(line, t):
        seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)]) / max(seg.sum(), 1e-9)
        return np.stack([np.interp(t, s, line[:, 0]), np.interp(t, s, line[:, 1])], 1)

    def world_lines(self):
        """All GT elements in world coordinates, for the global overlay and the offset estimate."""
        out = {c: [] for c in MAP_CLASSES}
        out['divider'] = [np.array(g.coords) for g in self.dividers]
        for name, geom in (('boundary', self.road_union), ('ped_crossing', ops.unary_union(self.crosswalks))):
            for poly in getattr(geom, 'geoms', [geom]):
                if poly.is_empty:
                    continue
                rings = [poly.exterior] + list(poly.interiors)
                if name == 'boundary':  # drop the map-end caps, as in frame_gt
                    out[name] += [np.array(g.coords) for r in rings for g in _lines(LineString(r.coords).difference(self.map_ends))]
                else:
                    out[name] += [np.array(r.coords) for r in rings]
        out['centerline'] = [np.array(ll['center'].coords) for ll in self.lanelets.values()]
        return out


# --------------------------------------------------------------------------- frames


class Frame:
    """Virtual nuScenes-LiDAR frame of one keyframe: origin 0.94 m ahead of the ego, x right, y forward."""

    def __init__(self, ego_xy, yaw):
        self.ego, self.yaw = np.asarray(ego_xy, float), yaw
        self.fwd = np.array([np.cos(yaw), np.sin(yaw)])
        self.right = np.array([np.sin(yaw), -np.cos(yaw)])
        self.o = self.ego + LIDAR_X_IN_EGO * self.fwd

    def to_local(self, geom):
        s, c = np.sin(self.yaw), np.cos(self.yaw)
        ox, oy = self.o
        return affinity.affine_transform(geom, [s, -c, c, s, -(ox * s - oy * c), -(ox * c + oy * s)])

    def to_world_pts(self, pts):
        pts = np.asarray(pts)
        return self.o + pts[:, :1] * self.right + pts[:, 1:2] * self.fwd

    def vicinity(self, r=40.0):
        return box(self.o[0] - r, self.o[1] - r, self.o[0] + r, self.o[1] + r)


def _lines(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == 'LineString':
        return [geom]
    if geom.geom_type == 'MultiLineString':
        return list(geom.geoms)
    if geom.geom_type == 'GeometryCollection':
        return [g for gg in geom.geoms for g in _lines(gg)]
    return []


def _contours(union, local_patch):
    """MapTR poly_geoms_to_instances / ped_poly_geoms_to_instances: oriented outline clipped to the patch."""
    if union.is_empty:
        return []
    if union.geom_type != 'MultiPolygon':
        union = MultiPolygon([union]) if union.geom_type == 'Polygon' else MultiPolygon([g for g in union.geoms if g.geom_type == 'Polygon'])
    out = []
    for poly in union.geoms:
        rings = [(poly.exterior, True)] + [(r, False) for r in poly.interiors]
        for ring, is_ext in rings:
            coords = list(ring.coords)
            if ring.is_ccw == is_ext:  # exteriors clockwise, interiors counter-clockwise
                coords = coords[::-1]
            lines = LineString(coords).intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            out += _lines(lines)
    return out


def _union_centerline(pieces, lanelets):
    """MapTRv2 union_centerline: chain lanelet centreline pieces through successors, return every root->leaf path."""
    G = nx.DiGraph()
    for lid, geom in pieces.items():
        parts = _lines(geom)
        pts_list = [np.array(p.coords).round(3) for p in parts]
        for pts in pts_list:
            for a, b in zip(pts[:-1], pts[1:]):
                G.add_edge(tuple(a), tuple(b))
        start, end = tuple(pts_list[0][0]), tuple(pts_list[-1][-1])
        for p in lanelets[lid]['pred']:
            if p in pieces:
                G.add_edge(tuple(np.array(_lines(pieces[p])[-1].coords).round(3)[-1]), start)
        for s in lanelets[lid]['succ']:
            if s in pieces:
                G.add_edge(end, tuple(np.array(_lines(pieces[s])[0].coords).round(3)[0]))
    roots = [v for v, d in G.in_degree() if d == 0]
    leaves = [v for v, d in G.out_degree() if d == 0]
    paths = []
    for r in roots:
        for path in nx.all_simple_paths(G, r, leaves):
            if len(path) > 1:
                paths.append(LineString(path).simplify(0.2, preserve_topology=True))
    return paths


def frame_gt(gt, fr):
    """GT instances of one frame in the model frame, as a MapTR annotation dict."""
    near = fr.vicinity()
    patch = box(PC_RANGE[0], PC_RANGE[1], PC_RANGE[3], PC_RANGE[4])
    hx, hy = PATCH_W / 2, PATCH_H / 2
    inst = {c: [] for c in MAP_CLASSES}

    for line in gt.dividers:
        if line.intersects(near):
            inst['divider'] += _lines(fr.to_local(line).intersection(patch))
    cw = [fr.to_local(p) for p in gt.crosswalks if p.intersects(near)]
    inst['ped_crossing'] = _contours(ops.unary_union(cw), box(-hx - 0.2, -hy - 0.2, hx + 0.2, hy + 0.2)) if cw else []
    roads = fr.to_local(gt.road_union.intersection(near))
    inst['boundary'] = _contours(roads, box(-hx + 0.2, -hy + 0.2, hx - 0.2, hy - 0.2))
    ends = gt.map_ends.intersection(near)
    if not ends.is_empty:
        ends = fr.to_local(ends)
        inst['boundary'] = [g for l in inst['boundary'] for g in _lines(l.difference(ends))]
    pieces = {}
    for lid, ll in gt.lanelets.items():
        if ll['center'].intersects(near):
            g = fr.to_local(ll['center']).intersection(patch)
            if _lines(g):
                pieces[lid] = g
    inst['centerline'] = _union_centerline(pieces, gt.lanelets) if pieces else []

    vectors = []
    for c, lines in inst.items():
        for l in lines:
            if l.length > 0.1:
                pts = np.array(l.coords)[:, :2]
                vectors.append(dict(pts=pts.round(4).tolist(), pts_num=len(pts), cls_name=c, type=MAP_CLASSES.index(c)))
    return vectors


# --------------------------------------------------------------------------- pose offset diagnostic


def estimate_offset(world_gt, frames_fr, preds, score_thr, search=3.0):
    """Constant world (dx, dy) that best aligns confident divider/boundary predictions with the map."""
    from scipy.spatial import cKDTree
    trees, pts = {}, {}
    for c in ('divider', 'boundary'):
        g = [resample(l, 0.2) for l in world_gt[c] if len(l) > 1]
        if g:
            trees[c] = cKDTree(np.concatenate(g))
        p = [resample(fr.to_world_pts(v['pts']), 0.5) for fr, res in zip(frames_fr, preds) for v in res['vectors']
             if v['cls_name'] == c and v['confidence_level'] >= thr_of(score_thr, c)]
        if p:
            pts[c] = np.concatenate(p)
    if not trees or not pts:
        return np.zeros(2), None, None

    def cost(d):
        tot = n = 0
        for c in pts:
            if c in trees:
                dist, _ = trees[c].query(pts[c] + d, distance_upper_bound=1.5)
                tot += np.minimum(dist, 1.5).sum()
                n += len(dist)
        return tot / max(n, 1)

    best = np.zeros(2)
    for step, span in ((0.5, search), (0.1, 0.5), (0.02, 0.1)):
        grid = np.arange(-span, span + 1e-9, step)
        cands = [best + [dx, dy] for dx in grid for dy in grid]
        best = min(cands, key=cost)
    return best, cost(np.zeros(2)), cost(best)


# --------------------------------------------------------------------------- scoring


def thr_of(score_thr, cls):
    """score_thr is a number, or a dict of per-class thresholds."""
    return score_thr[cls] if isinstance(score_thr, dict) else score_thr


def thr_text(score_thr):
    if isinstance(score_thr, dict):
        return 'score >= ' + ', '.join(f'{c} {t:g}' for c, t in score_thr.items())
    return f'score >= {score_thr:g}'


def score(results, annotations, out_dir, score_thr, nproc):
    """MapTR chamfer AP (0.5/1.0/1.5 m) + precision/recall at score_thr and 1.0 m."""
    from projects.mmdet3d_plugin.datasets.map_utils.mean_ap import eval_map, format_res_gt_by_classes
    from projects.mmdet3d_plugin.datasets.map_utils.tpfp import custom_tpfp_gen
    out_dir.mkdir(parents=True, exist_ok=True)
    cls_gens, cls_gts = format_res_gt_by_classes(str(out_dir / 'results.json'), results, annotations,
                                                 cls_names=MAP_CLASSES, num_pred_pts_per_instance=20,
                                                 eval_use_same_gt_sample_num_flag=True, pc_range=PC_RANGE, nproc=nproc)
    ap = {c: {} for c in MAP_CLASSES}
    num_gts = {}
    for thr in CHAMFER_THRESHOLDS:
        _, res = eval_map(results, annotations, cls_gens, cls_gts, threshold=thr, cls_names=MAP_CLASSES,
                          pc_range=PC_RANGE, metric='chamfer', num_pred_pts_per_instance=20, nproc=nproc)
        for c, r in zip(MAP_CLASSES, res):
            ap[c][str(thr)] = float(r['ap'])
            num_gts[c] = int(r['num_gts'])
    pr = {}
    for c in MAP_CLASSES:
        tp = fp = 0
        for gens, gts in zip(cls_gens[c], cls_gts[c]):
            gens = gens[gens[:, -1] >= thr_of(score_thr, c)]
            t, f = custom_tpfp_gen(gens, gts, threshold=1.0, metric='chamfer')
            tp, fp = tp + t.sum(), fp + f.sum()
        pr[c] = dict(precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(num_gts[c], 1)),
                     tp=int(tp), fp=int(fp), num_gt=num_gts[c])
    mean = {c: float(np.mean(list(ap[c].values()))) for c in MAP_CLASSES}
    valid = [mean[c] for c in MAP_CLASSES if num_gts[c] > 0]
    return dict(mAP=float(np.mean(valid)) if valid else 0.0, AP=mean, AP_by_threshold=ap,
                at_score_thr=dict(score_thr=score_thr, chamfer_m=1.0, per_class=pr), num_gt=num_gts)


# --------------------------------------------------------------------------- drawing


def local_to_px(pts, scale):
    """Model frame (x right, y fwd) -> pixel in the upscaled 300x150 ego map."""
    pts = np.asarray(pts)
    x_e, y_e = pts[:, 1] + LIDAR_X_IN_EGO, -pts[:, 0]
    return np.stack([(MAP_Y - y_e) / MAP_RES * scale, (MAP_X - x_e) / MAP_RES * scale], 1).round().astype(np.int32)


def draw_eval_frame(sem_map, front, gt_vecs, pred_vecs, title, scale=3):
    m = (0.45 * cv2.resize(sem_map, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)).astype(np.uint8)
    gt_layer = m.copy()
    for v in gt_vecs:
        cv2.polylines(gt_layer, [local_to_px(v['pts'], scale)], False, CLASS_BGR[v['type']], 7, cv2.LINE_AA)
    m = cv2.addWeighted(gt_layer, 0.45, m, 0.55, 0)
    for v in pred_vecs:
        px = local_to_px(v['pts'], scale)
        cv2.polylines(m, [px], False, CLASS_BGR[v['type']], 2, cv2.LINE_AA)
        cv2.circle(m, tuple(int(c) for c in px[-1]), 3, CLASS_BGR[v['type']], -1)
    c = local_to_px(np.array([[0.0, -LIDAR_X_IN_EGO]]), scale)[0]
    cv2.rectangle(m, (int(c[0]) - 5, int(c[1]) - 11), (int(c[0]) + 5, int(c[1]) + 4), (255, 255, 255), 1)
    cv2.putText(m, 'wide/faded = GT   thin = SemVecNet', (5, m.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for i, name in enumerate(MAP_CLASSES):
        cv2.putText(m, name, (5, 15 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLASS_BGR[i], 1, cv2.LINE_AA)
    if front is not None:
        f = cv2.resize(front, (int(front.shape[1] * m.shape[0] / front.shape[0]), m.shape[0]))
        m = np.hstack([f, m])
    cv2.putText(m, title, (5, 32 + 16 * len(MAP_CLASSES) if front is None else 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return m


def find_junctions(gt, merge=8.0):
    """Intersections of the map: areas where non-neighbouring road lanelets overlap (WorldGT.junctions), merged
    per intersection and named J1..Jn clockwise from north around the map centre."""
    if gt.junctions.is_empty:
        return []
    pieces = getattr(gt.junctions, 'geoms', [gt.junctions])
    merged = ops.unary_union([g.buffer(merge) for g in pieces]).buffer(-merge)
    polys = [p for p in getattr(merged, 'geoms', [merged]) if p.area > 20]
    mid = np.array(gt.road_union.centroid.coords[0])
    juncs = []
    for p in polys:
        c = np.array(p.centroid.coords[0])
        d = c - mid
        bearing = np.degrees(np.arctan2(d[0], d[1])) % 360  # clockwise from north
        juncs.append(dict(poly=p, center=c, bearing=bearing,
                          compass=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][int(((bearing + 22.5) % 360) // 45)]))
    juncs.sort(key=lambda j: j['bearing'])
    for i, j in enumerate(juncs):
        j['name'] = f'J{i + 1}'
    return juncs


def draw_global(world_gt, frames_fr, preds, score_thr, offset, path, res=0.1, margin=40.0, whole_map=False,
                bounds=None, title=None, max_px=8000):
    """Lanelet2 GT polylines (wide, faded) with every confident SemVecNet polyline on top, in UTM.
    whole_map=True frames the entire map (and the drive), bounds=(lo, hi) a given window, otherwise the drive."""
    traj = np.array([fr.ego for fr in frames_fr])
    if bounds is not None:
        lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    else:
        pts = np.concatenate([traj] + [l for ls in world_gt.values() for l in ls]) if whole_map else traj
        lo, hi = pts.min(0) - margin, pts.max(0) + margin
    W, H = (hi - lo) / res
    s = min(1.0, max_px / max(W, H))
    W, H = max(int(W * s), 1), max(int(H * s), 1)
    to_px = lambda p: np.stack([(p[:, 0] - lo[0]) / res * s, H - (p[:, 1] - lo[1]) / res * s], 1).round().astype(np.int32)
    canvas = np.full((H, W, 3), 20, np.uint8)
    gt_layer = canvas.copy()
    for i, c in enumerate(MAP_CLASSES):
        for l in world_gt[c]:
            cv2.polylines(gt_layer, [to_px(l + offset)], False, CLASS_BGR[i], max(2, int(0.6 / res * s)), cv2.LINE_AA)
    canvas = cv2.addWeighted(gt_layer, 0.5, canvas, 0.5, 0)
    for fr, r in zip(frames_fr, preds):
        for v in r['vectors']:
            if v['confidence_level'] >= thr_of(score_thr, v['cls_name']):
                cv2.polylines(canvas, [to_px(fr.to_world_pts(v['pts']))], False, CLASS_BGR[v['type']], 1, cv2.LINE_AA)
    cv2.polylines(canvas, [to_px(traj)], False, (255, 255, 255), 1, cv2.LINE_AA)
    if title:
        cv2.putText(canvas, title, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'wide/faded = Lanelet2 GT   thin = SemVecNet (%s)   white = ego path' % thr_text(score_thr),
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='output dir of stages 1 and 2')
    ap.add_argument('--osm', required=True, help='Lanelet2 map (.osm, lat/lon)')
    ap.add_argument('--epsg', type=int, default=None, help='CRS of the ego poses (default: UTM zone of the map)')
    ap.add_argument('--score-thr', type=float, default=0.4)
    ap.add_argument('--class-thr', default='', help='per-class thresholds overriding --score-thr for drawing and '
                    'precision/recall, e.g. divider=0.1 (AP always uses every prediction)')
    ap.add_argument('--map-buffer', type=float, default=3.0, help='metres around lanelets that count as mapped')
    ap.add_argument('--max-offmap', type=float, default=3.0, help='skip frames with the ego this far off the map')
    ap.add_argument('--pose-offset', type=float, nargs=2, default=None, metavar=('DX', 'DY'),
                    help='world shift added to the ego pose; default: estimated, reported, and applied only to '
                         'the "offset_corrected" scores')
    ap.add_argument('--junction-radius', type=float, default=25.0,
                    help='a frame counts for an intersection when the ego is within this many metres of it')
    ap.add_argument('--no-viz', action='store_true')
    ap.add_argument('--nproc', type=int, default=8)
    args = ap.parse_args()

    sys.path.insert(0, str(Path.cwd()))  # SemVecNet root, so `projects` is importable
    if args.class_thr:
        t = {c: args.score_thr for c in MAP_CLASSES}
        t.update({k: float(v) for k, v in (kv.split('=') for kv in args.class_thr.split(',') if kv)})
        args.score_thr = t
    run = Path(args.run)
    out = run / 'eval'
    out.mkdir(exist_ok=True)
    meta = json.loads((run / 'meta.json').read_text())
    frames = [json.loads(l) for l in (run / 'frames.jsonl').read_text().splitlines() if l.strip()]
    preds = json.loads((run / 'predictions.json').read_text())['results']
    by_token = {r['sample_token']: r for r in preds}
    if meta.get('origin') is None:
        sys.exit('meta.json has no pose origin; rerun stage 1')
    origin = np.array(meta['origin'][:2])

    lanelets, epsg = load_lanelet2(args.osm, args.epsg)
    gt = WorldGT(lanelets, args.map_buffer)
    world_lines = gt.world_lines()
    print(f'Map {args.osm}: {gt.n_road} road + {gt.n_cross} crosswalk lanelets, EPSG:{epsg}; '
          f'{len(gt.dividers)} divider pieces; poses from {meta.get("pose_source")} ({meta.get("ego_frame")})', flush=True)

    # keep frames on the map
    frs, used, skipped = [], [], 0
    for f in frames:
        fr = Frame(origin + np.array(f['pose'][:2]), f['pose'][2])
        if gt.road_union.distance(Point(*fr.ego)) > args.max_offmap or str(f['idx']) not in by_token:
            skipped += 1
            continue
        frs.append(fr)
        used.append(f)
    print(f'{len(used)} / {len(frames)} frames on the map ({skipped} skipped: ego > {args.max_offmap} m off the lanelets)')
    if not used:
        sys.exit('No frames on the map: is the bag driven on the mapped roads, and is the pose in UTM '
                 f'(EPSG:{epsg})? First ego position: {origin + np.array(frames[0]["pose"][:2])}')
    pred_list = [by_token[str(f['idx'])] for f in used]

    est, cost0, cost1 = estimate_offset(world_lines, frs, pred_list, args.score_thr)
    print(f'Pose/map offset estimate: dx={est[0]:+.2f} m dy={est[1]:+.2f} m '
          f'(mean distance of confident divider/boundary points to GT {cost0 or 0:.2f} -> {cost1 or 0:.2f} m)', flush=True)

    def mapped_only(fr, res):
        keep = []
        for v in res['vectors']:
            line = LineString(fr.to_world_pts(v['pts']))
            if line.length > 0 and line.intersection(gt.mapped).length / line.length >= 0.5:
                keep.append(v)
        return dict(res, vectors=keep)

    metrics = dict(map=args.osm, epsg=epsg, frames_scored=len(used), frames_skipped=skipped,
                   score_thr=args.score_thr, map_buffer_m=args.map_buffer,
                   pose_offset_estimate=[float(est[0]), float(est[1])])
    runs = [('raw', np.zeros(2) if args.pose_offset is None else np.array(args.pose_offset))]
    if args.pose_offset is None and np.linalg.norm(est) > 0.05:
        runs.append(('offset_corrected', est))
    gts_raw = res_mapped_raw = None
    for name, off in runs:
        frs_off = [Frame(fr.ego + off, fr.yaw) for fr in frs]
        annotations = [dict(sample_token=str(f['idx']), vectors=frame_gt(gt, fr)) for f, fr in zip(used, frs_off)]
        results_mapped = [mapped_only(fr, r) for fr, r in zip(frs_off, pred_list)]
        if gts_raw is None:
            gts_raw, res_mapped_raw = annotations, results_mapped
            (out / 'gt.json').write_text(json.dumps(dict(GTs=annotations)))
        metrics[name] = dict(
            pose_offset=[float(off[0]), float(off[1])],
            mapped=score(results_mapped, annotations, out / name / 'mapped', args.score_thr, args.nproc),
            full=score(pred_list, annotations, out / name / 'full', args.score_thr, args.nproc))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    if not args.no_viz:
        (out / 'viz').mkdir(exist_ok=True)
        writer = None
        for f, fr, r, g in zip(used, frs, pred_list, gts_raw):
            pv = [v for v in r['vectors'] if v['confidence_level'] >= thr_of(args.score_thr, v['cls_name'])]
            front = cv2.imread(str(run / f['front'])) if f.get('front') else None
            img = draw_eval_frame(cv2.imread(str(run / f['map'])), front, g['vectors'], pv, f'frame {f["idx"]}')
            cv2.imwrite(str(out / 'viz' / f'{f["idx"]:06d}.png'), img)
            if writer is None:
                writer = cv2.VideoWriter(str(out / 'eval.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                         float(f.get('fps', 2.0)), (img.shape[1], img.shape[0]))
            writer.write(img)
        if writer is not None:
            writer.release()
        draw_global(world_lines, frs, pred_list, args.score_thr, np.zeros(2), out / 'global_overlay.png',
                    margin=15.0, whole_map=True)
        draw_global(world_lines, frs, pred_list, args.score_thr, np.zeros(2), out / 'global_overlay_drive.png')

    # ---------------- per intersection: frames with the ego within --junction-radius of it
    juncs = find_junctions(gt)
    jdir = out / 'junctions'
    jdir.mkdir(exist_ok=True)
    dist = np.array([[j['poly'].distance(Point(*fr.ego)) for j in juncs] for fr in frs]).reshape(len(frs), len(juncs))
    near = dist <= args.junction_radius
    jmetrics, tiles = {}, []
    for k, j in enumerate(juncs):
        idx = np.where(near[:, k])[0]
        entry = dict(compass=j['compass'], center_utm=[round(float(v), 2) for v in j['center']], frames=int(len(idx)))
        if len(idx):
            entry.update(score([res_mapped_raw[i] for i in idx], [gts_raw[i] for i in idx],
                               out / 'raw' / 'junctions' / j['name'], args.score_thr, args.nproc))
            if not args.no_viz:
                c = j['center']
                title = f'{j["name"]} ({j["compass"]})  {len(idx)} frames  mAP {entry["mAP"]:.3f}'
                draw_global(world_lines, [frs[i] for i in idx], [pred_list[i] for i in idx], args.score_thr,
                            np.zeros(2), jdir / f'{j["name"]}.png', res=0.05, bounds=(c - 40, c + 40), title=title)
                closest = used[idx[np.argmin(dist[idx, k])]]
                frame_img = cv2.imread(str(out / 'viz' / f'{closest["idx"]:06d}.png'))
                if frame_img is not None:
                    cv2.imwrite(str(jdir / f'{j["name"]}_closest_frame.png'), frame_img)
                tiles.append(cv2.resize(cv2.imread(str(jdir / f'{j["name"]}.png')), (800, 800), interpolation=cv2.INTER_AREA))
        jmetrics[j['name']] = entry
    far = np.where(~near.any(1))[0] if len(juncs) else np.arange(len(frs))
    metrics['raw']['junctions'] = jmetrics
    if len(far):
        metrics['raw']['away_from_junctions'] = dict(frames=int(len(far)), **score(
            [res_mapped_raw[i] for i in far], [gts_raw[i] for i in far], out / 'raw' / 'away', args.score_thr, args.nproc))
    if tiles:
        cols = min(3, len(tiles))
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
        cv2.imwrite(str(out / 'junctions.png'), np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    print('\n' + '=' * 78)
    print(f'SemVecNet vs {Path(args.osm).name}: {len(used)} frames, chamfer AP (mean of 0.5/1.0/1.5 m)')
    for name, _ in runs:
        for region in ('mapped', 'full'):
            m = metrics[name][region]
            row = '  '.join(f'{c} {m["AP"][c]:.3f}' for c in MAP_CLASSES)
            print(f'  {name:16s} {region:6s}  mAP {m["mAP"]:.3f}   {row}')
    m = metrics['raw']['mapped']['at_score_thr']['per_class']
    print(f'  raw/mapped @{thr_text(args.score_thr)}, 1.0 m: ' +
          '  '.join(f'{c} P {m[c]["precision"]:.2f} R {m[c]["recall"]:.2f} (GT {m[c]["num_gt"]})' for c in MAP_CLASSES))
    print(f'\n  Intersections (raw, mapped; ego within {args.junction_radius:.0f} m):')
    rows = list(metrics['raw']['junctions'].items())
    if 'away_from_junctions' in metrics['raw']:
        rows.append(('away', dict(compass='-', **metrics['raw']['away_from_junctions'])))
    for name, e in rows:
        if not e['frames']:
            print(f'    {name:5s} {e["compass"]:3s} not driven')
            continue
        pr = e['at_score_thr']['per_class']
        print(f'    {name:5s} {e["compass"]:3s} {e["frames"]:4d} frames  mAP {e["mAP"]:.3f}  ' +
              '  '.join(f'{c} AP {e["AP"][c]:.2f} P {pr[c]["precision"]:.2f} R {pr[c]["recall"]:.2f}' for c in MAP_CLASSES))
    print('=' * 78)
    print(f'Wrote {out / "metrics.json"}' + ('' if args.no_viz else
          f', {out / "eval.mp4"}, {out / "global_overlay.png"} (whole map), {out / "global_overlay_drive.png"}'))


if __name__ == '__main__':
    main()
