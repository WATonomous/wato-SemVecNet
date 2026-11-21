#!/usr/bin/env python3
"""Side-by-side view of one run of run_mcap_pipeline.sh, one image per keyframe + a video:

    +------------------ the raw camera images of the keyframe (surround order) ------------------+
    | LiDAR sweep | semantic map (SemVecNet's input) | SemVecNet output | ground truth | overlay   |

All bird's-eye panels are the same 60 x 30 m window around the ego (rear axle), forward = up, left = left.
  LiDAR          the keyframe's sweep, coloured by intensity (lane paint is bright)
  semantic map   built by stage 1 from the cameras (HRNet) and the LiDAR, accumulated over time
  SemVecNet      predicted polylines with score >= --score-thr
  ground truth   the Lanelet2 map cut to this keyframe (eval/gt.json, from tools/mcap_eval_lanelet2.py)
  overlay        ground truth wide and faded, SemVecNet thin on top

    python tools/mcap_compare_viz.py --run <run dir>     ->  <run>/compare/<frame>.png, <run>/compare.mp4
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MAP_CLASSES = ['divider', 'ped_crossing', 'boundary', 'centerline']
CLASS_BGR = [(0, 165, 255), (255, 80, 0), (0, 0, 255), (0, 200, 0)]
LIDAR_X_IN_EGO = 0.943713
MAP_RES, MAP_X, MAP_Y = 0.2, 30.0, 15.0          # m/px of the semantic map, half length / width of the window
SCALE = 2.4                                      # panel px per semantic-map px -> 360 x 720 panels
PW, PH = int(2 * MAP_Y / MAP_RES * SCALE), int(2 * MAP_X / MAP_RES * SCALE)
BG = (24, 24, 24)


def ego_to_px(xy):
    """Ego (x fwd, y left) metres -> panel pixels."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    return np.stack([(MAP_Y - xy[:, 1]) / MAP_RES * SCALE, (MAP_X - xy[:, 0]) / MAP_RES * SCALE], 1)


def model_to_ego(pts):
    """SemVecNet / MapTR frame (x right, y fwd, origin 0.94 m ahead of the ego) -> ego frame."""
    pts = np.asarray(pts, float)
    return np.stack([pts[:, 1] + LIDAR_X_IN_EGO, -pts[:, 0]], 1)


def blank():
    p = np.full((PH, PW, 3), BG, np.uint8)
    for m in range(-30, 31, 10):  # 10 m grid
        y = int(ego_to_px([[m, 0]])[0, 1])
        cv2.line(p, (0, y), (PW, y), (45, 45, 45), 1)
    for m in (-10, 0, 10):
        x = int(ego_to_px([[0, m]])[0, 0])
        cv2.line(p, (x, 0), (x, PH), (45, 45, 45), 1)
    return p


def ego_marker(p):
    c = ego_to_px([[0, 0]])[0].astype(int)  # rear axle; car body ~4.5 x 1.9 m around it
    a, b = ego_to_px([[3.6, 0.95]])[0].astype(int), ego_to_px([[-0.9, -0.95]])[0].astype(int)
    cv2.rectangle(p, tuple(int(v) for v in a), tuple(int(v) for v in b), (255, 255, 255), 1)
    cv2.circle(p, tuple(int(v) for v in c), 2, (255, 255, 255), -1)
    return p


def draw_lines(p, vectors, width, alpha=1.0):
    layer = p.copy()
    for v in vectors:
        px = ego_to_px(model_to_ego(v['pts'])).round().astype(np.int32)
        col = CLASS_BGR[v['type']]
        cv2.polylines(layer, [px], False, col, width, cv2.LINE_AA)
        if width <= 3:
            cv2.circle(layer, tuple(int(c) for c in px[-1]), 3, col, -1)  # end point = direction
    return cv2.addWeighted(layer, alpha, p, 1 - alpha, 0) if alpha < 1 else layer


def lidar_panel(path):
    p = blank()
    if path is None or not path.exists():
        return p
    pts = np.load(path).astype(np.float32)
    px = ego_to_px(pts[:, :2]).astype(int)
    ok = (px[:, 0] >= 0) & (px[:, 0] < PW) & (px[:, 1] >= 0) & (px[:, 1] < PH)
    px, inten = px[ok], pts[ok, 3]
    if not len(inten):
        return p
    hi = max(float(np.percentile(inten, 99)), 1.0)
    v = np.sqrt(np.clip(inten / hi, 0, 1))  # sqrt: most returns are dim, keep them visible
    col = cv2.applyColorMap((40 + 215 * v).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0]
    for i in np.argsort(inten):  # bright returns (paint) drawn last
        cv2.circle(p, (int(px[i, 0]), int(px[i, 1])), 1, tuple(int(c) for c in col[i]), -1)
    return p


def label(p, title, sub=''):
    cv2.rectangle(p, (0, 0), (p.shape[1], 40 if sub else 24), (0, 0, 0), -1)
    cv2.putText(p, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(p, sub, (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return p


def camera_grid(frame, run, azimuths, width):
    """Raw camera images: forward-facing row (left -> right) over rear-facing row (left -> right)."""
    cams = frame.get('inputs', {}).get('cams', {})
    if not cams:
        return None
    az = {c: azimuths.get(c, 0.0) for c in cams}
    front = sorted([c for c in cams if np.cos(np.radians(az[c])) >= 0], key=lambda c: -az[c])
    back = sorted([c for c in cams if np.cos(np.radians(az[c])) < 0], key=lambda c: az[c] % 360)
    rows = [r for r in (front, back) if r]
    tw = width // max(len(r) for r in rows)
    out = []
    for r in rows:
        tiles = []
        for c in r:
            im = cv2.imread(str(run / cams[c]))
            im = cv2.resize(im, (tw, int(tw * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
            name = c.strip('/').split('/')[0]
            cv2.rectangle(im, (0, 0), (len(name) * 9 + 10, 20), (0, 0, 0), -1)
            cv2.putText(im, name, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(im)
        row = np.hstack(tiles)
        out.append(cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT, value=BG))
    return np.vstack(out)


def legend(width):
    bar = np.full((30, width, 3), BG, np.uint8)
    x = 10
    for i, c in enumerate(MAP_CLASSES):
        cv2.line(bar, (x, 15), (x + 30, 15), CLASS_BGR[i], 3)
        cv2.putText(bar, c, (x + 38, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        x += 80 + 10 * len(c)
    cv2.putText(bar, 'dot = polyline end (direction)   grid = 10 m   box = car', (x + 20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
    return bar


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--score-thr', type=float, default=0.4)
    ap.add_argument('--only-gt', action='store_true', help='only frames that have ground truth (on the map)')
    ap.add_argument('--class-thr', default='', help='per-class score thresholds overriding --score-thr, '
                    'e.g. divider=0.1,ped_crossing=0.3')
    ap.add_argument('--out-name', default='compare', help='output folder / video name inside the run')
    args = ap.parse_args()

    run = Path(args.run)
    meta = json.loads((run / 'meta.json').read_text())
    frames = [json.loads(l) for l in (run / 'frames.jsonl').read_text().splitlines() if l.strip()]
    preds = {r['sample_token']: r for r in json.loads((run / 'predictions.json').read_text())['results']}
    gt_path = run / 'eval' / 'gt.json'
    gts = {g['sample_token']: g for g in json.loads(gt_path.read_text())['GTs']} if gt_path.exists() else {}
    azimuths = meta.get('camera_azimuth_deg', {})
    thr = {c: args.score_thr for c in MAP_CLASSES}
    thr.update({k: float(v) for k, v in (kv.split('=') for kv in args.class_thr.split(',') if kv)})
    thr_txt = ', '.join(f'{c[:4]} {thr[c]:g}' for c in MAP_CLASSES) if args.class_thr else f'score >= {args.score_thr}'
    (run / args.out_name).mkdir(exist_ok=True)

    width = 5 * PW
    writer, size, n = None, None, 0
    for f in frames:
        tok = str(f['idx'])
        if args.only_gt and tok not in gts:
            continue
        pv = [v for v in preds.get(tok, {}).get('vectors', []) if v['confidence_level'] >= thr[MAP_CLASSES[v['type']]]]
        gv = gts.get(tok, {}).get('vectors')

        lidar = run / f['inputs']['lidar'] if 'inputs' in f else None
        p_lidar = label(ego_marker(lidar_panel(lidar)), 'INPUT  LiDAR sweep', 'coloured by intensity')
        sem = cv2.resize(cv2.imread(str(run / f['map'])), (PW, PH), interpolation=cv2.INTER_NEAREST)
        p_sem = label(ego_marker(sem), 'SemVecNet input: semantic map', 'HRNet(cameras) x LiDAR, accumulated')
        p_pred = label(ego_marker(draw_lines(blank(), pv, 2)), 'OUTPUT  SemVecNet',
                       f'{len(pv)} polylines, {thr_txt}')
        if gv is None:
            p_gt = label(ego_marker(blank()), 'GROUND TRUTH  Lanelet2', 'not on the mapped road')
            p_ov = label(ego_marker(draw_lines(blank(), pv, 2)), 'OVERLAY', 'no ground truth here')
        else:
            p_gt = label(ego_marker(draw_lines(blank(), gv, 2)), 'GROUND TRUTH  Lanelet2', f'{len(gv)} polylines')
            p_ov = label(ego_marker(draw_lines(draw_lines(blank(), gv, 9, alpha=0.45), pv, 2)), 'OVERLAY',
                         'GT wide/faded, SemVecNet thin')
        bev = np.hstack([p_lidar, p_sem, p_pred, p_gt, p_ov])
        for k in range(1, 5):
            cv2.line(bev, (k * PW, 0), (k * PW, PH), (90, 90, 90), 1)

        head = np.zeros((34, width, 3), np.uint8)
        t = (f['stamp_ns'] - frames[0]['stamp_ns']) / 1e9
        cv2.putText(head, f'frame {f["idx"]}   t = {t:6.1f} s   stamp {f["stamp_ns"]}   UTM E {meta["origin"][0] + f["pose"][0]:.1f} '
                    f'N {meta["origin"][1] + f["pose"][1]:.1f}   heading {np.degrees(f["pose"][2]):.0f} deg   {f.get("cams", 0)} cameras',
                    (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cams = camera_grid(f, run, azimuths, width)
        img = np.vstack([head] + ([cams] if cams is not None else []) + [legend(width), bev])
        cv2.imwrite(str(run / args.out_name / f'{f["idx"]:06d}.png'), img)
        if writer is None:
            size = (img.shape[1], img.shape[0])
            writer = cv2.VideoWriter(str(run / f'{args.out_name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), float(f.get('fps', 2.0)), size)
        writer.write(img if (img.shape[1], img.shape[0]) == size else cv2.resize(img, size))
        n += 1
    if writer is not None:
        writer.release()
    print(f'Wrote {n} frames to {run / args.out_name} and {run / (args.out_name + ".mp4")}')


if __name__ == '__main__':
    main()
