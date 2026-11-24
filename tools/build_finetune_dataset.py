#!/usr/bin/env python3
"""Fine-tuning dataset from stage-1 runs: semantic maps + Lanelet2 ground-truth polylines, split into train / test.

  python tools/build_finetune_dataset.py --runs RUNS_DIR --osm ring_road.osm --out data --test 'recording_20260224_213947_*'

Input: the run directories of tools/batch_stage1.py (maps/, frames.jsonl, meta.json, offset/offset.json, DONE).
For every run, keyframes are filtered and thinned:
  warmup        the first --warmup s of every continuous stretch (the map has not built up yet)
  off map       the ego further than --max-offmap m from a road lanelet
  alignment     the keyframe's paint/curb fit after the offset correction below --min-fit (mcap_estimate_offset.py)
  unmapped      more than --max-unmapped of the road the semantic map sees lies outside the mapped area: the map
                would label real road (side streets, parking lots) as "nothing"
  no GT         no polyline in the patch
  spacing       one keyframe per --spacing m of driving (60 m maps: consecutive kept maps overlap <= 50%), every
                --dense-spacing m within --dense-radius m of a junction or crosswalk and on tight curves (roundabout)
Ground truth is cut out of the map at the offset-corrected pose exactly as tools/mcap_eval_lanelet2.py does
(MapTRv2's own GT construction), in the model frame: x right, y forward, metres, origin 0.944 m ahead of the
rear axle (nuScenes LIDAR_TOP).

Split: whole drives are assigned to train or test after filtering. By default a deterministic greedy assignment
targets --test-fraction in every light / snow / camera-set stratum. --test remains an explicit whole-drive override.

Output (--out):
  manifest.json                 settings, per-run counts (kept / dropped per reason), totals
  train/ test/
    maps/<run>/<idx>.png         300 x 150 semantic map (model input, copied from the run)
    labels/<run>/<idx>.json      polylines + pose, offset, alignment fit, camera set, light (day / dusk / night)
    train.pkl / test.pkl         the same for mmdet3d (RingRoadSemanticMapDataset), paths relative to the split dir
  previews/train, previews/test  a sample of maps with their polylines drawn on
"""
import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import math
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_common import WorldRaster, class_masks, ego_pose_world, load_run, pixels_to_world  # noqa: E402
from filter_dataset import uncovered_road  # noqa: E402
from mcap_eval_lanelet2 import (LIDAR_X_IN_EGO, MAP_CLASSES, Frame, WorldGT, find_junctions,  # noqa: E402
                                load_lanelet2, frame_gt)

CLASS_BGR = [(0, 165, 255), (255, 0, 0), (0, 0, 255), (0, 200, 0)]  # divider, ped_crossing, boundary, centerline
MODEL_FRAME = 'x right, y forward, m; origin 0.944 m ahead of the rear axle (nuScenes LIDAR_TOP), as MapTRv2 predicts'
# recordings dated December - March (Waterloo winter: snow banks along the Ring Road, which HRNet labels as lane
# paint and sidewalk, so those semantic maps are too messy to learn from)
DEFAULT_SNOW_PATTERNS = ['*20[0-9][0-9]12[0-3][0-9]_*', '*20[0-9][0-9]0[1-3][0-3][0-9]_*']


def solar_elevation(stamp_ns, lat, lon):
    t = dt.datetime.fromtimestamp(stamp_ns / 1e9, dt.timezone.utc)
    g = 2 * math.pi / 365 * (t.timetuple().tm_yday - 1 + (t.hour + t.minute / 60 - 12) / 24)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                    - 0.040849 * math.sin(2 * g))
    ha = math.radians(((t.hour * 60 + t.minute + t.second / 60) + eqt + 4 * lon) / 4 - 180)
    la = math.radians(lat)
    return math.degrees(math.asin(math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)))


def light(stamp_ns, lat, lon):
    e = solar_elevation(stamp_ns, lat, lon)
    return 'day' if e > 6 else 'dusk' if e > -6 else 'night'


def camera_set(run_dir):
    done = run_dir / 'DONE'
    if done.exists():
        return json.loads(done.read_text()).get('camera_set')
    return run_dir.name.rsplit('__', 1)[-1] if '__' in run_dir.name else None


def check_image(run_dir, frame, sem_bgr, vectors, title, scale=3):
    """One image to verify a training map: the cameras (all of them for new runs, else the front camera with its
    segmentation) | the semantic map | the map dimmed with the ground-truth polylines drawn on."""
    sem = cv2.resize(sem_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    with_gt = draw_preview(sem_bgr, vectors, '', scale)
    h = sem.shape[0]
    panels = []
    cam_rel = frame.get('cams_all') or frame.get('front')
    cam = cv2.imread(str(run_dir / cam_rel)) if cam_rel else None
    if cam is not None:
        panels.append(cv2.resize(cam, (int(cam.shape[1] * h / cam.shape[0]), h), interpolation=cv2.INTER_AREA))
    sep = np.full((h, 6, 3), 255, np.uint8)
    for p in (sem, with_gt):
        panels += [sep, p] if panels else [p]
    img = np.hstack(panels)
    bar = np.zeros((26, img.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    x = img.shape[1] - 4 * 120
    for name, bgr in zip(MAP_CLASSES, CLASS_BGR):  # legend for the polyline colours
        cv2.line(bar, (x, 13), (x + 18, 13), bgr, 3)
        cv2.putText(bar, name, (x + 22, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        x += 120
    return np.vstack([bar, img])


def draw_preview(sem_bgr, vectors, title, scale=3):
    m = cv2.resize(sem_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    m = (0.6 * m).astype(np.uint8)
    for v in vectors:
        p = np.asarray(v['pts'], float)
        ex, ey = p[:, 1] + LIDAR_X_IN_EGO, -p[:, 0]  # model frame -> ego (x fwd, y left)
        px = np.stack([(15.0 - ey) / 0.2 * scale, (30.0 - ex) / 0.2 * scale], 1).round().astype(np.int32)
        cv2.polylines(m, [px], False, CLASS_BGR[MAP_CLASSES.index(v['cls'])], 2, cv2.LINE_AA)
    cv2.putText(m, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return m


def junction_leg_count(gt, junction, radius=35.0, min_radius=10.0, angle_tol=25.0):
    """Count road arms at a find_junctions() polygon by clustering nearby lanelet-centreline bearings."""
    center = junction['center']
    bearings = []
    core = junction['poly'].buffer(min_radius)
    for lanelet in gt.lanelets.values():
        line = lanelet['center']
        if line.distance(junction['poly']) > radius:
            continue
        for point in np.asarray(line.coords):
            delta = point - center
            distance = np.linalg.norm(delta)
            if min_radius <= distance <= radius and not core.contains(Point(*point)):
                bearings.append(math.degrees(math.atan2(delta[1], delta[0])) % 360)
    clusters = []
    for bearing in sorted(bearings):
        if not any(abs((bearing - other + 180) % 360 - 180) <= angle_tol for other in clusters):
            clusters.append(bearing)
    if len(clusters) > 1 and abs((clusters[0] - clusters[-1] + 180) % 360 - 180) <= angle_tol:
        clusters.pop()
    return len(clusters)


def nearest_junction(ego, junctions, radius):
    nearby = [(np.linalg.norm(ego - j['center']), j) for j in junctions]
    if not nearby:
        return None
    distance, junction = min(nearby, key=lambda item: item[0])
    return junction if distance < radius else None


def snow_for(run_name, patterns):
    return any(fnmatch.fnmatch(run_name, pattern) for pattern in patterns)


def select(run_dir, gt, dense_geoms, junctions, args):
    """Kept keyframes of one run: [(frame, corrected ego xy, yaw, offset, fit, vectors)], plus drop counts."""
    meta, frames = load_run(run_dir)
    origin = np.array(meta['origin'][:2], float)
    oj = run_dir / 'offset' / 'offset.json'
    offs = json.loads(oj.read_text())['frames'] if oj.exists() and not args.no_offset else {}
    drops = dict(warmup=0, off_map=0, alignment=0, junction_missing_road=0, unmapped=0, no_gt=0, spacing=0)
    if not frames:
        return [], drops

    ego = np.array([ego_pose_world(f, origin)[0] for f in frames])
    off = np.array([offs.get(str(f['idx']), {}).get('offset', [0.0, 0.0]) for f in frames]) if offs else np.zeros((len(frames), 2))
    ego_c = ego + off
    yaw = np.array([f['pose'][2] for f in frames])
    t = np.array([f['stamp_ns'] for f in frames]) / 1e9
    # continuous stretches (stage 1 --near-map leaves gaps where the car was away from the map)
    seg_start = t.copy()
    for i in range(1, len(t)):
        seg_start[i] = seg_start[i - 1] if t[i] - t[i - 1] < 2.5 else t[i]
    step = np.r_[0, np.linalg.norm(np.diff(ego, axis=0), axis=1)]
    step[np.r_[False, np.diff(t) >= 2.5]] = 0
    dist = np.cumsum(step)
    # tight curves: heading change per metre over the neighbouring +-10 m of driving
    curv = np.zeros(len(frames))
    for i in range(len(frames)):
        a = np.searchsorted(dist, dist[i] - 10)
        b = min(np.searchsorted(dist, dist[i] + 10, side='right') - 1, len(frames) - 1)
        if dist[b] - dist[a] > 5:
            dyaw = np.angle(np.exp(1j * (yaw[b] - yaw[a])))
            curv[i] = abs(dyaw) / (dist[b] - dist[a])

    lo, hi = ego_c.min(0) - 45, ego_c.max(0) + 45
    raster = WorldRaster(lo, hi, res=0.5)
    mapped = raster.fill(gt.mapped).astype(np.float32)

    kept, last = [], -1e9
    for i, f in enumerate(frames):
        if t[i] - seg_start[i] < args.warmup:
            drops['warmup'] += 1
            continue
        if gt.road_union.distance(Point(*ego_c[i])) > args.max_offmap:
            drops['off_map'] += 1
            continue
        rec = offs.get(str(f['idx']), {})
        fit = rec.get('fit')
        if offs and args.min_fit > 0 and (fit is None or fit < args.min_fit or rec.get('points', 0) < args.min_points):
            drops['alignment'] += 1
            continue
        sem = cv2.imread(str(run_dir / f['map']))
        junction = nearest_junction(ego_c[i], junctions, args.dense_radius)
        if junction is not None:
            pose = [ego_c[i][0], ego_c[i][1], yaw[i]]
            _, patches = uncovered_road(sem, pose, raster, mapped)
            if any(area >= args.missing_road_min_area and reach >= args.missing_road_min_reach
                   for area, reach in patches):
                drops['junction_missing_road'] += 1
                continue
        dense = curv[i] > 1.0 / args.curve_radius or any(g.distance(Point(*ego_c[i])) < args.dense_radius for g in dense_geoms)
        if dist[i] - last < (args.dense_spacing if dense else args.spacing):
            drops['spacing'] += 1
            continue
        road = class_masks(sem)['road']
        f_c = dict(f, pose=[ego_c[i][0] - origin[0], ego_c[i][1] - origin[1], yaw[i]])
        pts = pixels_to_world(road, f_c, origin, stride=3)
        if len(pts) and (raster.sample(mapped, pts, outside=0) < 0.5).mean() > args.max_unmapped:
            drops['unmapped'] += 1
            continue
        vectors = [dict(cls=v['cls_name'], pts=v['pts']) for v in frame_gt(gt, Frame(ego_c[i], yaw[i]))]
        if not vectors:
            drops['no_gt'] += 1
            continue
        kept.append((f, ego_c[i], float(yaw[i]), off[i], fit, dense, junction['legs'] if junction else None, vectors))
        last = dist[i]
    return kept, drops


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', required=True, help='directory of tools/batch_stage1.py run directories')
    ap.add_argument('--osm', required=True)
    ap.add_argument('--out', required=True, help='dataset directory (train/, test/, previews/, manifest.json)')
    ap.add_argument('--test', nargs='*', default=[], help='run-name patterns of the held-out drive(s)')
    ap.add_argument('--test-fraction', type=float, default=0.2,
                    help='target test fraction of final maps when --test is not supplied (default: 0.2)')
    ap.add_argument('--snow', nargs='*', default=DEFAULT_SNOW_PATTERNS,
                    help='run-name patterns recorded with snow (default: recordings dated December - March)')
    ap.add_argument('--keep-snow', action='store_true', help='export snow runs too (default: leave them out)')
    ap.add_argument('--test-cameras', default='pano8',
                    help='deprecated compatibility option; whole drives now stay in one split')
    # 5 m: snow-free Ring Road data is scarce, so neighbouring keyframes overlap heavily (training sees them as
    # augmented views); the per-drive train/test split keeps that overlap out of the test score
    ap.add_argument('--spacing', type=float, default=5.0, help='m of driving between kept keyframes')
    ap.add_argument('--dense-spacing', type=float, default=3.0, help='m between kept keyframes near junctions, '
                    'crosswalks and on tight curves')
    ap.add_argument('--dense-radius', type=float, default=30.0, help='m from a junction / crosswalk that counts as near')
    ap.add_argument('--curve-radius', type=float, default=40.0, help='curves tighter than this radius (m) count as dense')
    ap.add_argument('--warmup', type=float, default=6.0, help='s dropped at the start of every continuous stretch')
    ap.add_argument('--max-offmap', type=float, default=3.0)
    ap.add_argument('--min-fit', type=float, default=0.5, help='keyframe alignment fit needed (0 = keep all)')
    ap.add_argument('--min-points', type=int, default=40)
    ap.add_argument('--max-unmapped', type=float, default=0.25)
    ap.add_argument('--missing-road-min-area', type=float, default=60.0,
                    help='m2 uncovered observed road that marks a missing junction arm')
    ap.add_argument('--missing-road-min-reach', type=float, default=8.0,
                    help='m an uncovered patch must extend beyond mapped road')
    ap.add_argument('--no-offset', action='store_true', help='ignore offset/offset.json (raw GNSS poses)')
    ap.add_argument('--previews', type=int, default=24, help='preview images per split')
    ap.add_argument('--no-check', action='store_true', help='skip check/<split>/<run>/<idx>.jpg (cameras | semantic '
                    'map | map with ground-truth polylines, one per kept map)')
    ap.add_argument('--dry-run', action='store_true', help='report selections and split counts without writing --out')
    args = ap.parse_args()
    if not 0 < args.test_fraction < 1:
        ap.error('--test-fraction must be between 0 and 1')

    runs_dir, out = Path(args.runs), Path(args.out)
    runs = sorted(p for p in runs_dir.iterdir() if (p / 'DONE').exists() and (p / 'frames.jsonl').exists())
    if not args.keep_snow:
        snowy = [r.name for r in runs if snow_for(r.name, args.snow)]
        if snowy:
            print(f'Leaving out {len(snowy)} snow runs: {", ".join(snowy)}', flush=True)
        runs = [r for r in runs if r.name not in snowy]
    if not runs:
        sys.exit(f'No finished runs (DONE) in {runs_dir}')
    lanelets, epsg = load_lanelet2(args.osm)
    gt = WorldGT(lanelets, map_buffer=3.0)
    junctions = find_junctions(gt)
    for junction in junctions:
        junction['legs'] = junction_leg_count(gt, junction)
    dense_geoms = [j['poly'] for j in junctions] + list(gt.crosswalks)
    from pyproj import Transformer
    c = np.array(gt.road_union.centroid.coords[0])
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(c[0], c[1])

    # Select first: both the split target and strata are based on maps that survive all filters.
    selected = {}
    run_meta = {}
    for run in runs:
        meta, _ = load_run(run)
        selected[run] = select(run, gt, dense_geoms, junctions, args)
        run_meta[run] = meta

    by_drive = {}
    for run, (kept, _) in selected.items():
        by_drive.setdefault(run.name.rsplit('__', 1)[0], []).append(run)

    def run_strata(run):
        camera = camera_set(run)
        return Counter((light(f['stamp_ns'], lat, lon), 'snow' if snow_for(run.name, args.snow) else 'no snow', camera)
                       for f, *_ in selected[run][0])

    drive_strata = {drive: sum((run_strata(run) for run in drive_runs), Counter())
                    for drive, drive_runs in by_drive.items()}
    all_strata = sum(drive_strata.values(), Counter())
    target = {key: count * args.test_fraction for key, count in all_strata.items()}
    test_drives = set()
    if args.test:
        test_drives = {drive for drive in by_drive
                       if any(fnmatch.fnmatch(drive, pattern) or
                              any(fnmatch.fnmatch(run.name, pattern) for run in drive_runs)
                              for pattern in args.test)}
    else:
        # Largest drives first minimizes indivisible-drive overshoot; hash breaks ties independently of directory order.
        order = sorted(by_drive, key=lambda drive: (-sum(drive_strata[drive].values()),
                                                     hashlib.sha256(drive.encode()).hexdigest()))
        assigned = Counter()
        def score(counts):
            # Weight by available maps, not by the tiny per-stratum target: otherwise a one-map stratum can
            # dominate the decision and turn a nominal 20% holdout into a much larger one.
            strata_error = sum((counts[key] - target[key]) ** 2 / max(all_strata[key], 1) for key in target)
            overall_error = (sum(counts.values()) - sum(all_strata.values()) * args.test_fraction) ** 2 / \
                            max(sum(all_strata.values()), 1)
            return strata_error + overall_error
        for drive in order:
            with_drive = assigned + drive_strata[drive]
            if score(with_drive) < score(assigned):
                test_drives.add(drive)
                assigned = with_drive
        if not test_drives and order:
            # A drive can be larger than its stratum target; still provide a meaningful held-out set.
            test_drives.add(min(order, key=lambda drive: abs(sum(drive_strata[drive].values()) -
                                                               sum(all_strata.values()) * args.test_fraction)))

    def split_of(run):
        return 'test' if run.name.rsplit('__', 1)[0] in test_drives else 'train'

    if not args.dry_run:
        for sub in ('train', 'test', 'previews', 'check', 'manifest.json'):  # only what this tool writes
            p = out / sub
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
    infos = dict(train=[], test=[])
    manifest = dict(created=dt.datetime.now().isoformat(timespec='seconds'), osm=Path(args.osm).name, epsg=epsg,
                    model_frame=MODEL_FRAME, map_classes=MAP_CLASSES,
                    settings={k: v for k, v in vars(args).items() if k not in ('runs', 'out')}, runs={})
    for run in runs:
        split = split_of(run)
        meta = run_meta[run]
        kept, drops = selected[run]
        cams = len(meta.get('cameras', []))
        for f, ego_c, yaw, off, fit, dense, junction_legs, vectors in kept:
            idx = f'{f["idx"]:06d}'
            lt = light(f['stamp_ns'], lat, lon)
            rel_map = f'maps/{run.name}/{idx}.png'
            label = dict(token=f'{run.name}_{idx}', run=run.name, bag=[Path(b).name for b in meta.get('bag', [])],
                          keyframe=f['idx'], stamp_ns=f['stamp_ns'], camera_set=camera_set(run), cameras=cams,
                          light=lt, snow=snow_for(run.name, args.snow), dense_area=bool(dense),
                          junction_legs=junction_legs,
                          pose_utm=[round(float(ego_c[0] - off[0]), 3), round(float(ego_c[1] - off[1]), 3), round(yaw, 5)],
                          offset_m=[round(float(off[0]), 3), round(float(off[1]), 3)],
                          pose_utm_corrected=[round(float(ego_c[0]), 3), round(float(ego_c[1]), 3), round(yaw, 5)],
                          alignment_fit=fit, frame=MODEL_FRAME, map=rel_map,
                          vectors=[dict(cls=v['cls'], pts=np.round(v['pts'], 3).tolist()) for v in vectors])
            if not args.dry_run:
                (out / split / 'maps' / run.name).mkdir(parents=True, exist_ok=True)
                (out / split / 'labels' / run.name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(run / f['map'], out / split / rel_map)
                (out / split / 'labels' / run.name / f'{idx}.json').write_text(json.dumps(label))
                if not args.no_check:
                    (out / 'check' / split / run.name).mkdir(parents=True, exist_ok=True)
                    title = (f'{run.name} #{f["idx"]}  {split}  {lt}{"  snow" if label["snow"] else ""}  {cams} cams  '
                             f'fit {"-" if fit is None else round(fit, 2)}'
                             f'{f"  {junction_legs}-way junction" if junction_legs else ""}')
                    cv2.imwrite(str(out / 'check' / split / run.name / f'{idx}.jpg'),
                                check_image(run, f, cv2.imread(str(run / f['map'])), label['vectors'], title),
                                [cv2.IMWRITE_JPEG_QUALITY, 80])
            infos[split].append(dict(
                token=label['token'], scene_token=run.name, frame_idx=len(infos[split]), timestamp=f['stamp_ns'] / 1e9,
                sem_path=rel_map, ego_xy=np.array(ego_c, np.float64), yaw=yaw, camera_set=label['camera_set'],
                cameras=cams, light=lt, snow=label['snow'], junction_legs=junction_legs,
                annotation={c: [np.asarray(v['pts'], np.float32) for v in vectors if v['cls'] == c] for c in MAP_CLASSES}))
        manifest['runs'][run.name] = dict(split=split, keyframes=sum(drops.values()) + len(kept), kept=len(kept),
                                          dropped=drops, cameras=cams, camera_set=camera_set(run),
                                          light=sorted({light(f['stamp_ns'], lat, lon) for f, *_ in kept}))
        print(f'  {split:5s} {run.name:48s} kept {len(kept):4d} / {manifest["runs"][run.name]["keyframes"]:4d}  '
              f'dropped {drops}', flush=True)

    if not args.dry_run:
        for split, lst in infos.items():
            if not lst:
                continue
            with open(out / split / f'{split}.pkl', 'wb') as fh:
                pickle.dump(dict(infos=lst, metadata=dict(version='ringroad-v1', map_classes=MAP_CLASSES,
                                                           model_frame=MODEL_FRAME)), fh, protocol=4)
            sample = [lst[i] for i in np.linspace(0, len(lst) - 1, min(args.previews, len(lst))).round().astype(int)]
            (out / 'previews' / split).mkdir(parents=True, exist_ok=True)
            for info in sample:
                lab = json.loads((out / split / 'labels' / (info['sem_path'][5:-4] + '.json')).read_text())
                img = draw_preview(cv2.imread(str(out / split / info['sem_path'])), lab['vectors'],
                                   f'{lab["run"]} #{lab["keyframe"]} {lab["light"]}')
                cv2.imwrite(str(out / 'previews' / split / f'{info["token"]}.png'), img)
    manifest['totals'] = {s: dict(maps=len(l), runs=len({i['scene_token'] for i in l}),
                                  by_light={k: sum(i['light'] == k for i in l) for k in ('day', 'dusk', 'night')},
                                   by_snow={k: sum(bool(i['snow']) == (k == 'snow') for i in l)
                                            for k in ('snow', 'no snow')},
                                   by_camera_set={k: sum(i['camera_set'] == k for i in l) for k in sorted({i['camera_set'] for i in l})},
                                   near_junction={f'{k}-way': sum(i['junction_legs'] == k for i in l) for k in (3, 4)},
                                   vectors={c: sum(len(i['annotation'][c]) for i in l) for c in MAP_CLASSES})
                           for s, l in infos.items()}
    manifest['split'] = dict(method='explicit --test whole-drive override' if args.test else 'deterministic greedy',
                             test_fraction=args.test_fraction, test_drives=sorted(test_drives),
                             strata_target={' / '.join(map(str, k)): round(v, 2) for k, v in target.items()},
                             junctions=[dict(name=j['name'], legs=j['legs']) for j in junctions])
    if not args.dry_run:
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    for s, tot in manifest['totals'].items():
        print(f'{s}: {tot["maps"]} maps from {tot["runs"]} runs; light {tot["by_light"]}; snow {tot["by_snow"]}; '
              f'cameras {tot["by_camera_set"]}; junctions {tot["near_junction"]}')
    if not infos['test']:
        print('WARNING: empty test set - pass --test with the held-out drive', flush=True)
    print('Dry run only; no files written.' if args.dry_run else f'Dataset in {out}')


if __name__ == '__main__':
    main()
