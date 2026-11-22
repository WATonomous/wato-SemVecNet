#!/usr/bin/env python3
"""Copy a fine-tuning dataset (tools/build_finetune_dataset.py) without the maps whose labels contradict what the
semantic map shows: road the car saw that ring_road.osm does not have (a missing junction leg, side street or
parking-lot entrance) gets no labels, so training on it teaches the model to draw nothing on real road.

  python tools/filter_dataset.py --src ringroad_dataset_large --osm ring_road.osm --out ringroad_dataset_clean \
      [--exclude-junction J1 --exclude-radius 40] [--min-area 60 --min-reach 8] [--report-only]

A map is dropped when
  junction   the ego is within --exclude-radius m of a named junction of the OSM (tools/mcap_eval_lanelet2.py
             find_junctions names them J1..Jn; J1 is the 4-way junction that the OSM maps as a T)
  missing    a patch of observed road (road / crosswalk in the semantic map) lies outside the mapped area
             (lanelets + 3 m) with >= --min-area m2 that reaches >= --min-reach m away from it: a road leg, not a
             parking lane or a curb sliver next to a mapped road
Everything else is kept unchanged (maps, labels, pkl entries), so the dataset stays as large as possible.

Output: the same layout as the source (train/ test/ maps, labels, <split>.pkl, manifest.json with the filter
settings and per-map reasons) plus dropped/<split>/<reason>/<token>.png: the map with the uncovered road in yellow
and its labels, to check what was removed.
"""
import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_common import MAP_RES, WorldRaster, class_masks, pixels_to_world  # noqa: E402
from mcap_eval_lanelet2 import MAP_CLASSES, WorldGT, find_junctions, load_lanelet2  # noqa: E402

CLASS_BGR = {'divider': (0, 165, 255), 'ped_crossing': (255, 80, 0), 'boundary': (0, 0, 255), 'centerline': (0, 200, 0)}


def uncovered_road(sem, pose, raster, mapped):
    """Observed road outside the mapped area: (mask, [(area m2, reach m) per patch])."""
    m = class_masks(sem)
    road = m['road'] | m['crosswalk']
    allpts = pixels_to_world(np.ones(road.shape, bool), dict(pose=pose), np.zeros(2))
    inside = (raster.sample(mapped, allpts, outside=0) >= 0.5).reshape(road.shape)
    un = cv2.morphologyEx((road & ~inside).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    reach = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 5) * MAP_RES  # m to the mapped area
    n, lab, stats, _ = cv2.connectedComponentsWithStats(un, 8)
    patches = [(float(stats[k, cv2.CC_STAT_AREA] * MAP_RES ** 2), float(reach[lab == k].max())) for k in range(1, n)]
    return un, patches


def preview(sem, un, vectors, title):
    k = 3
    a = cv2.resize(sem, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
    b = a.copy()
    m = cv2.resize(un * 255, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST) > 0
    b[m] = (0.4 * b[m] + 0.6 * np.array([0, 255, 255])).astype(np.uint8)
    for v in vectors:
        p = np.array(v['pts'])
        px = np.stack([(5 * p[:, 0] + 75) * k, (-5 * p[:, 1] + (30 - 0.943713) * 5) * k], 1).round().astype(np.int32)
        cv2.polylines(b, [px], False, CLASS_BGR[v['cls']], 2, cv2.LINE_AA)
    img = np.hstack([a, b])
    cv2.putText(img, title, (5, 18), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, type=Path)
    ap.add_argument('--osm', required=True)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--exclude-junction', nargs='*', default=['J1'], help='OSM junction names to drop (J1 J2 ...)')
    ap.add_argument('--exclude-radius', type=float, default=40.0)
    ap.add_argument('--min-area', type=float, default=60.0, help='m2 of uncovered road that counts as a missing road')
    ap.add_argument('--min-reach', type=float, default=8.0, help='m the patch must reach beyond the mapped area')
    ap.add_argument('--report-only', action='store_true', help='print what would be dropped, write nothing')
    args = ap.parse_args()
    if not args.report_only and not args.out:
        ap.error('--out is required unless --report-only')
    if args.out and args.out.resolve() == args.src.resolve():
        ap.error('--out must differ from --src')

    lanelets, _ = load_lanelet2(args.osm)
    gt = WorldGT(lanelets, map_buffer=3.0)
    juncs = {j['name']: j for j in find_junctions(gt)}
    unknown = [n for n in args.exclude_junction if n not in juncs]
    if unknown:
        sys.exit(f'unknown junction(s) {unknown}; the OSM has {sorted(juncs)}')
    for n in args.exclude_junction:
        print(f'excluding {n} at UTM {juncs[n]["center"].round(1).tolist()} (radius {args.exclude_radius} m)')

    data = {s: pickle.load(open(args.src / s / f'{s}.pkl', 'rb')) for s in ('train', 'test')}
    labels = {s: {i['token']: json.loads((args.src / s / 'labels' / (i['sem_path'][5:-4] + '.json')).read_text())
                  for i in d['infos']} for s, d in data.items()}
    poses = np.array([l['pose_utm_corrected'][:2] for ls in labels.values() for l in ls.values()])
    raster = WorldRaster(poses.min(0) - 50, poses.max(0) + 50, res=0.5)
    mapped = raster.fill(gt.mapped).astype(np.float32)

    if args.out and not args.report_only:
        if args.out.exists():
            shutil.rmtree(args.out)
        args.out.mkdir(parents=True)
    manifest = json.loads((args.src / 'manifest.json').read_text())
    manifest['filter'] = dict(source=str(args.src), settings={k: str(v) if isinstance(v, Path) else v
                                                               for k, v in vars(args).items()}, dropped={})
    totals = {}
    for split, d in data.items():
        keep, reasons = [], {}
        for info in d['infos']:
            lab = labels[split][info['token']]
            ego = np.array(lab['pose_utm_corrected'][:2])
            reason = next((f'junction_{n}' for n in args.exclude_junction
                           if np.linalg.norm(ego - juncs[n]['center']) < args.exclude_radius), None)
            sem = cv2.imread(str(args.src / split / info['sem_path']))
            un, patches = uncovered_road(sem, lab['pose_utm_corrected'], raster, mapped)
            legs = [(a, r) for a, r in patches if a >= args.min_area and r >= args.min_reach]
            if reason is None and legs:
                reason = 'missing_road'
            if reason is None:
                keep.append(info)
                continue
            reasons[info['token']] = dict(reason=reason, patches=[[round(a), round(r, 1)] for a, r in legs])
            if not args.report_only:
                big = max(legs, default=(0, 0))
                img = preview(sem, un, lab['vectors'], f'{split} {info["token"][-36:]}  {reason}  '
                                                       f'{big[0]:.0f} m2, reaches {big[1]:.1f} m')
                (args.out / 'dropped' / split / reason).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.out / 'dropped' / split / reason / f'{info["token"]}.png'), img)
        by = {}
        for r in reasons.values():
            by[r['reason']] = by.get(r['reason'], 0) + 1
        print(f'{split}: keep {len(keep)} / {len(d["infos"])}  dropped {by}')
        manifest['filter']['dropped'][split] = reasons
        totals[split] = dict(maps=len(keep), runs=len({i['scene_token'] for i in keep}),
                             by_light={k: sum(i['light'] == k for i in keep) for k in ('day', 'dusk', 'night')},
                             vectors={c: sum(len(i['annotation'][c]) for i in keep) for c in MAP_CLASSES})
        if args.report_only:
            continue
        for info in keep:
            rel_label = 'labels/' + info['sem_path'][5:-4] + '.json'
            for rel in (info['sem_path'], rel_label):
                (args.out / split / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.src / split / rel, args.out / split / rel)
        keep = [dict(i, frame_idx=k) for k, i in enumerate(keep)]
        with open(args.out / split / f'{split}.pkl', 'wb') as fh:
            pickle.dump(dict(infos=keep, metadata=d['metadata']), fh, protocol=4)
    if not args.report_only:
        manifest['totals_before_filter'] = manifest.get('totals')
        manifest['totals'] = totals
        (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
        print(f'Wrote {args.out}; check what was dropped in {args.out / "dropped"}')


if __name__ == '__main__':
    main()
