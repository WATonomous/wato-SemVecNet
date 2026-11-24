#!/usr/bin/env python3
"""Pose/map offset of a stage-1 run, measured from the semantic maps themselves (no model predictions).

The ground truth for fine-tuning is cut out of the Lanelet2 map at the GNSS pose of every keyframe. Any offset
between that pose and the map (GNSS error, map georeferencing error) shifts every label by the same amount, so it
has to be removed before training. tools/mcap_eval_lanelet2.py also estimates an offset, but from SemVecNet's own
predictions - the thing being fine-tuned - and it changes by more than a metre between runs of the same bag.

Here the evidence is what stage 1 observed, placed in the world with the GNSS pose:
  lane paint            should lie on lanelet bounds              (weight 1.0, distance capped at 1.0 m)
  crosswalk paint       should lie inside crosswalk lanelets      (weight 1.0, capped at 1.5 m)
  curb (road|sidewalk)  should lie on the road boundary           (weight 1.0, capped at 1.5 m)
  road                  should lie inside the road lanelets       (weight 0.5, capped at 1.5 m)
  sidewalk              should lie outside the road lanelets      (weight 0.5, capped at 1.5 m)
The offset (dx, dy) minimises the weighted mean capped distance: first one offset for the whole run (coarse-to-fine
grid search, +-3 m), then one per time window (default 30 s), pulled towards the run offset so that windows with
little evidence (a straight road only constrains the lateral direction) do not wander. Keyframes get their offset
interpolated between window centres.

Convention: the corrected ego position is pose + offset (same as --pose-offset of mcap_eval_lanelet2.py).

Output in <run>/offset/
  offset.json   run offset, per-window offsets, per-keyframe offsets, alignment before/after, verdict
  overlay.png   whole run: map (grey lanelet bounds, red road boundary, blue crosswalks), lane paint before (orange)
                and after (green) the correction
  zoom.png      four close-ups along the run, before | after, with lane paint (white), crosswalk paint (cyan),
                curbs (magenta)
Check overlay.png / zoom.png once per run: after the correction the painted lines should sit on the grey lines.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_common import (MAP_RES, MAP_X, MAP_Y, WorldRaster, class_masks, curb_mask, dedupe,  # noqa: E402
                             ego_pose_world, load_run, paint_mask, pixels_to_world)
from mcap_eval_lanelet2 import WorldGT, load_lanelet2  # noqa: E402

# evidence class: (weight, cap m, pixel stride)
# Painted lines pin the position down; the road / sidewalk edges of the smoothed semantic maps (stage 1 renders with
# the paper's 3x3 filter) sit ~0.5-1 m off the map's boundaries and pulled the offsets away from the paint.
TERMS = dict(lane=(1.0, 1.0, 1), crosswalk=(1.0, 1.5, 2), curb=(0.3, 1.5, 1), road=(0.0, 1.5, 3), sidewalk=(0.0, 1.5, 2))
LINE_TERMS = ('lane', 'crosswalk', 'curb')  # the ones that pin the position down; road/sidewalk only help
MIN_PTS = 50
LANE_TOL, CURB_TOL = 0.4, 0.5  # m: a paint / curb point "sits on" a map line within this distance (keyframe fit)
# edge lines are painted inside the curb, which the map's outer lanelet bounds follow: paint within this distance of
# the road boundary also counts as on a map line (at night HRNet often sees only the edge lines)
EDGE_TOL = 0.8
VERSION = 3  # bump when the fit changes, so run_phase1.py recomputes older offset.json files
MIN_FRAME_PTS = 40             # nearby paint + curb points a keyframe needs for its fit to mean anything


def near_mask(fwd, lat):
    """Pixels within fwd m ahead/behind and lat m left/right of the ego: the camera ground labels are only accurate
    close to the car (a small pitch error moves paint 25 m ahead by a metre)."""
    v = np.arange(int(2 * MAP_X / MAP_RES))[:, None]
    u = np.arange(int(2 * MAP_Y / MAP_RES))[None, :]
    return (np.abs(MAP_X - v * MAP_RES) <= fwd) & (np.abs(MAP_Y - u * MAP_RES) <= lat)


def collect(run, frames, origin, gt, max_offmap, near=(12.0, 10.0)):
    """Per keyframe: world points of every evidence class (keyframes with the ego far off the map are skipped)."""
    out = []
    keep = near_mask(*near)
    for f in frames:
        ego, _ = ego_pose_world(f, origin)
        if gt.road_union.distance(Point(*ego)) > max_offmap:
            continue
        img = cv2.imread(str(run / f['map']))
        if img is None:
            continue
        m = class_masks(img)
        m['curb'] = curb_mask(m)
        m['lane'] = paint_mask(m)
        m = {c: v & keep for c, v in m.items()}
        out.append((f, {c: pixels_to_world(m[c], f, origin, TERMS[c][2]) for c in TERMS}))
    return out


def group(items):
    return {c: dedupe(np.concatenate([p[c] for _, p in items])) if items else np.zeros((0, 2)) for c in TERMS}


def distance_fields(raster, gt, world_lines):
    bounds = [ll[s] for ll in gt.lanelets.values() for s in ('left', 'right')]
    road = raster.fill(gt.road_union)
    cross = raster.blank()
    for p in gt.crosswalks:
        raster.fill(p, cross)
    return dict(lane=raster.distance(raster.draw_lines(bounds)),
                crosswalk=raster.distance(cross),
                curb=raster.distance(raster.draw_lines(world_lines['boundary'])),
                road=raster.distance(road),
                sidewalk=raster.distance(1 - road))


def make_cost(raster, fields, groups):
    use = {c: p for c, p in groups.items() if len(p) >= MIN_PTS}

    def cost(d):
        tot = wsum = 0.0
        for c, p in use.items():
            w, cap, _ = TERMS[c]
            tot += w * np.minimum(raster.sample(fields[c], p + d), cap).mean()
            wsum += w
        return tot / wsum if wsum else np.nan
    return cost


def search(cost, center, levels):
    best = np.asarray(center, float)
    for step, span in levels:
        g = np.arange(-span, span + 1e-9, step)
        cands = [best + np.array([dx, dy]) for dx in g for dy in g]
        vals = np.array([cost(c) for c in cands])
        if np.all(np.isnan(vals)):
            return best
        best = cands[int(np.nanargmin(vals))]
    return best


def keep_mapped(raster, mapped, groups):
    """Drop evidence outside the mapped area (+ buffer): parking lots, side streets and their paint have no map
    element to align with and only add noise."""
    return {c: p[raster.sample(mapped, p, outside=0) > 0.5] for c, p in groups.items()}


def fit(raster, field, pts, d, near=1.0, tight=0.3):
    """Alignment of paint with map lines: median distance of the points within `near` m of a line (inliers),
    and the fraction of all points within `tight` m."""
    if len(pts) < MIN_PTS:
        return None
    v = raster.sample(field, pts + d)
    inl = v[v <= near]
    return dict(inlier_median_m=round(float(np.median(inl)), 3) if len(inl) else None,
                within_m=tight, fraction_within=round(float((v <= tight).mean()), 3))


# ----------------------------------------------------------------------------------------------- drawing


def draw(lo, res, size, gt, world_lines, layers, traj=None):
    """World -> image (north up). layers: list of (points, BGR)."""
    W, H = size
    img = np.full((H, W, 3), 25, np.uint8)

    def px(p):
        p = (np.asarray(p, float) - lo) / res
        return np.stack([p[:, 0], H - 1 - p[:, 1]], 1).round().astype(np.int32)

    for ll in gt.lanelets.values():
        for s in ('left', 'right'):
            cv2.polylines(img, [px(ll[s])], False, (110, 110, 110), 1, cv2.LINE_AA)
    for l in world_lines['boundary']:
        cv2.polylines(img, [px(l)], False, (0, 0, 220), 1, cv2.LINE_AA)
    for l in world_lines['ped_crossing']:
        cv2.polylines(img, [px(l)], False, (220, 90, 0), 1, cv2.LINE_AA)
    if traj is not None and len(traj) > 1:
        cv2.polylines(img, [px(traj)], False, (255, 255, 255), 1, cv2.LINE_AA)
    for pts, color in layers:
        if len(pts):
            q = px(pts)
            ok = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
            img[q[ok, 1], q[ok, 0]] = color
    return img


def draw_outputs(out, gt, world_lines, items, groups, off_run, traj):
    lo, hi = traj.min(0) - 35, traj.max(0) + 35
    res = max(0.25, float((hi - lo).max()) / 6000)
    size = tuple(int(v) for v in np.ceil((hi - lo) / res))
    img = draw(lo, res, size, gt, world_lines,
               [(groups['lane'], (0, 140, 255)), (groups['lane'] + off_run, (0, 230, 0))], traj)
    cv2.putText(img, f'lane paint before (orange) / after (green) offset {off_run[0]:+.2f} {off_run[1]:+.2f} m',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out / 'overlay.png'), img)

    rows = []
    for k in np.linspace(0, len(items) - 1, min(4, len(items))).round().astype(int):
        f, p = items[k]
        c = p['road'].mean(0) if len(p['road']) else None
        if c is None:
            continue
        lo_c, res_c, n = c - 30, 0.1, 600
        panels = []
        for d, title in ((np.zeros(2), 'before'), (f['offset'], 'after')):
            pan = draw(lo_c, res_c, (n, n), gt, world_lines,
                       [(p['curb'] + d, (255, 0, 255)), (p['crosswalk'] + d, (255, 255, 0)), (p['lane'] + d, (255, 255, 255))])
            cv2.putText(pan, f'frame {f["idx"]} {title} ({d[0]:+.2f}, {d[1]:+.2f}) m', (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(pan)
        rows.append(np.hstack(panels))
    if rows:
        cv2.imwrite(str(out / 'zoom.png'), np.vstack(rows))


# ----------------------------------------------------------------------------------------------- main


def estimate(run, osm, window_s, prior, max_offmap, search_m, min_fit, gt_cache={}):
    run = Path(run)
    meta, frames = load_run(run)
    origin = np.array(meta['origin'][:2], float)
    if osm not in gt_cache:
        lanelets, epsg = load_lanelet2(osm)
        gt = WorldGT(lanelets, map_buffer=3.0)
        gt_cache[osm] = (gt, gt.world_lines(), epsg)
    gt, world_lines, epsg = gt_cache[osm]
    if 'novatel' not in str(meta.get('pose_source', '')):
        print(f'  WARNING: pose source {meta.get("pose_source")} is not NovAtel GNSS odometry; '
              'the poses may not be in the map CRS at all', flush=True)

    items = collect(run, frames, origin, gt, max_offmap)
    if len(items) < 5:
        print(f'  {run.name}: only {len(items)} keyframes within {max_offmap} m of the map - nothing to align', flush=True)
        return None
    groups = group(items)
    pts_all = np.concatenate([g for g in groups.values() if len(g)])
    raster = WorldRaster(pts_all.min(0) - search_m - 5, pts_all.max(0) + search_m + 5)
    fields = distance_fields(raster, gt, world_lines)
    mapped = raster.fill(gt.mapped).astype(np.float32)
    groups = keep_mapped(raster, mapped, groups)
    items = [(f, keep_mapped(raster, mapped, p)) for f, p in items]

    cost = make_cost(raster, fields, groups)
    c0 = cost(np.zeros(2))
    off_run = search(cost, np.zeros(2), ((0.25, search_m), (0.05, 0.3), (0.01, 0.06)))
    c1 = cost(off_run)

    # windows, regularised towards the run offset
    t = np.array([f['stamp_ns'] for f, _ in items]) / 1e9
    edges = np.arange(t[0], t[-1] + window_s, window_s)
    windows = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = [it for it, ti in zip(items, t) if a <= ti < b]
        if not sel:
            continue
        g = group(sel)
        n_line = sum(len(g[c]) for c in LINE_TERMS)
        wc = make_cost(raster, fields, g)
        if n_line < 6 * MIN_PTS:
            off, fallback = off_run, True
        else:
            off = search(lambda d: wc(d) + prior * float(np.sum((d - off_run) ** 2)), off_run,
                         ((0.25, search_m), (0.05, 0.3), (0.01, 0.06)))
            fallback = False
        windows.append(dict(t_start=float(a - t[0]), t_end=float(b - t[0]), frames=len(sel), lane_points=len(g['lane']),
                            line_points=n_line, offset=[round(float(v), 3) for v in off], fallback_to_run=fallback,
                            cost_before=round(float(wc(np.zeros(2))), 4), cost_after=round(float(wc(off)), 4)))

    centres = np.array([(w['t_start'] + w['t_end']) / 2 for w in windows]) + t[0]
    offs = np.array([w['offset'] for w in windows])
    per_frame = {}
    for f in frames:
        ts = f['stamp_ns'] / 1e9
        per_frame[str(f['idx'])] = [round(float(np.interp(ts, centres, offs[:, k])), 3) for k in range(2)]
    for f, _ in items:
        f['offset'] = np.array(per_frame[str(f['idx'])])

    # per-keyframe alignment after its own offset: share of its nearby lane paint / curb points that sit on a map line
    frame_fit = {}
    for f, p in items:
        rec = dict(offset=per_frame[str(f['idx'])])
        for key, d in (('fit_before', np.zeros(2)), ('fit', f['offset'])):
            ok = n = 0
            for c, tol in (('lane', LANE_TOL), ('curb', CURB_TOL)):
                if len(p[c]):
                    on = raster.sample(fields[c], p[c] + d) <= tol
                    if c == 'lane':
                        on |= raster.sample(fields['curb'], p[c] + d) <= EDGE_TOL
                    ok += int(on.sum())
                    n += len(p[c])
            rec[key] = round(ok / n, 3) if n else None
        rec['points'] = n
        frame_fit[str(f['idx'])] = rec
    for f in frames:  # keyframes off the map: offset only, no fit (the dataset export drops them anyway)
        frame_fit.setdefault(str(f['idx']), dict(offset=per_frame[str(f['idx'])], fit=None, fit_before=None, points=0))
    usable = [r for r in frame_fit.values() if r['fit'] is not None and r['fit'] >= min_fit and r['points'] >= MIN_FRAME_PTS]
    share = len(usable) / max(len(items), 1)

    lane = {k: fit(raster, fields['lane'], groups['lane'], d) for k, d in (('before', np.zeros(2)), ('after', off_run))}
    curb = {k: fit(raster, fields['curb'], groups['curb'], d, near=1.5, tight=0.5)
            for k, d in (('before', np.zeros(2)), ('after', off_run))}
    spread = float(np.max(np.linalg.norm(offs - off_run, axis=1))) if len(offs) else 0.0
    verdict = 'good' if share >= 0.6 else 'check' if share >= 0.3 else 'bad'

    out = run / 'offset'
    out.mkdir(exist_ok=True)
    result = dict(version=VERSION, run=run.name, osm=str(osm), epsg=epsg, pose_source=meta.get('pose_source'),
                  run_offset=[round(float(v), 3) for v in off_run], verdict=verdict,
                  cost_before=round(float(c0), 4), cost_after=round(float(c1), 4),
                  lane_paint_fit=lane, curb_fit=curb,
                  max_window_deviation_m=round(spread, 3), window_s=window_s, prior=prior,
                  keyframes_used=len(items), keyframes_total=len(frames),
                  usable=dict(min_fit=min_fit, min_points=MIN_FRAME_PTS, keyframes=len(usable), share=round(share, 3)),
                  points={c: int(len(p)) for c, p in groups.items()}, windows=windows, frames=frame_fit)
    (out / 'offset.json').write_text(json.dumps(result, indent=1))
    traj = np.array([ego_pose_world(f, origin)[0] for f, _ in items])
    draw_outputs(out, gt, world_lines, items, groups, off_run, traj)
    fmt = lambda r: '-' if r is None else f'{r["inlier_median_m"]} m, {100 * r["fraction_within"]:.0f}% within {r["within_m"]} m'
    print(f'  {run.name}: offset {off_run[0]:+.2f} {off_run[1]:+.2f} m  [{verdict}]\n'
          f'      lane paint to lanelet bounds: {fmt(lane["before"])}  ->  {fmt(lane["after"])}\n'
          f'      curbs to road boundary:       {fmt(curb["before"])}  ->  {fmt(curb["after"])}\n'
          f'      window offsets up to {spread:.2f} m from the run offset; {len(items)}/{len(frames)} keyframes on the map, '
          f'{len(groups["lane"])} lane / {len(groups["curb"])} curb / {len(groups["crosswalk"])} crosswalk points\n'
          f'      usable for training (fit >= {min_fit}): {len(usable)}/{len(items)} keyframes ({100 * share:.0f}%)',
          flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', nargs='+', required=True, help='stage-1 run directories')
    ap.add_argument('--osm', required=True, help='Lanelet2 map, e.g. ring_road.osm')
    ap.add_argument('--window', type=float, default=20.0, help='seconds per window offset (default 20)')
    ap.add_argument('--prior', type=float, default=0.05, help='pull of window offsets towards the run offset, per m^2')
    ap.add_argument('--max-offmap', type=float, default=10.0, help='skip keyframes with the ego this far off the map')
    ap.add_argument('--search', type=float, default=3.0, help='search range of the offsets, +- m')
    ap.add_argument('--min-fit', type=float, default=0.5, help='keyframe fit (share of nearby paint/curb points on a '
                    'map line after correction) needed to count as usable for training')
    args = ap.parse_args()
    bad = 0
    for r in args.run:
        res = estimate(r, args.osm, args.window, args.prior, args.max_offmap, args.search, args.min_fit)
        bad += res is None or res['verdict'] != 'good'
    if bad:
        print(f'{bad} run(s) need a look: open <run>/offset/overlay.png and zoom.png', flush=True)


if __name__ == '__main__':
    main()
