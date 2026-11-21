#!/usr/bin/env python3
"""Stage 2 of the MCAP pipeline: SemVecNet's own semantic BEV maps -> vector map polylines.

Input  (written by tools/mcap_semantic_mapping.py)
  <run>/frames.jsonl   one JSON object per frame: idx, stamp_ns, map, front, pose=[x, y, yaw] (map frame)
  <run>/maps/*.png     300x150 ego-centric semantic maps (top = forward, image-left = ego-left, 0.2 m/px)

Output
  <run>/vectors/<idx>.json   polylines (score >= --score-thr) in the ego frame: x forward, y left, metres
  <run>/predictions.json     every prediction of every frame in the model's own frame (MapTR result format,
                             x right, y forward, origin 0.94 m ahead of the ego), for tools/mcap_eval_lanelet2.py
  <run>/viz/<idx>.png        front camera | semantic map with predicted polylines
  <run>/semvecnet.mp4        the viz frames as a video
  <run>/global_vectors.png   every frame's polylines placed in the map frame with the ego trajectory

The semantic map is loaded exactly like the test pipeline in semvecnet_nusc_centerline.py
(mmcv.imread -> NormalizeSemanticMap(to_rgb=True)), and the checkpoint is used unmodified.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

MAP_CLASSES = ['divider', 'ped_crossing', 'boundary', 'centerline']
CLASS_BGR = [(0, 165, 255), (255, 0, 0), (0, 0, 255), (0, 200, 0)]  # orange, blue, red, green
# SemVecNet (MapTRv2 nuScenes) predicts in the nuScenes LIDAR_TOP frame: x = right, y = forward.
# Its semantic map is centred on the ego (rear axle), so the LiDAR origin sits this far ahead of it.
LIDAR_X_IN_EGO = 0.943713
MAP_RES, MAP_X, MAP_Y = 0.2, 30.0, 15.0  # m/px, half-length forward, half-width lateral


def build_model(config, checkpoint, device):
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    cfg = Config.fromfile(config)
    sys.path.insert(0, str(Path.cwd()))  # SemVecNet root, so `projects` is importable
    importlib.import_module(cfg.plugin_dir.strip('/').replace('/', '.'))  # register plugin modules
    from mmdet3d.models import build_model as _build
    cfg.model.pretrained = dict(sem=None)  # weights come from the checkpoint below
    cfg.model.train_cfg = None
    model = _build(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint, map_location='cpu', strict=True)
    norm = next(t for t in cfg.data.test.pipeline if t['type'] == 'NormalizeSemanticMap')
    return model.to(device).eval(), norm


def load_map(path, norm):
    """Same ops as CustomLoadSemanticMapFromFile + NormalizeSemanticMap."""
    import mmcv
    img = mmcv.imread(str(path), 'unchanged').astype(np.float32)
    return mmcv.imnormalize(img, np.array(norm['mean'], np.float32), np.array(norm['std'], np.float32),
                            norm.get('to_rgb', True))


@torch.no_grad()
def predict(model, maps, device):
    x = torch.from_numpy(np.stack(maps)).permute(0, 3, 1, 2).contiguous().to(device)
    meta = [dict(scene_token='mcap', can_bus=np.zeros(18), pad_shape=[m.shape], img_shape=[m.shape]) for m in maps]
    _, results = model.simple_test(meta, sem_map=x)
    return [(r['pts_bbox']['pts_3d'].numpy(), r['pts_bbox']['scores_3d'].numpy(),
             r['pts_bbox']['labels_3d'].numpy()) for r in results]


LANE_RGB = (255, 255, 255)  # "lane" (painted markings) in the semantic map, semantic_mapping base_cfg LABEL_COLORS


def paint_support(pts, sem_map_bgr, radius_m=0.6, step_m=0.5):
    """How much of a predicted divider is backed by lane paint in the semantic map it was predicted from.

    Returns (observed, support): the fraction of the line over mapped (non-black) cells, and of that part the
    fraction within radius_m of lane-paint cells. SemVecNet places dividers from the road's shape (a habit from
    nuScenes, where paint rarely made it into the map), so on this car it can invent lanes with high confidence.
    """
    lane = np.all(sem_map_bgr == LANE_RGB[::-1], axis=2).astype(np.uint8)
    r = max(1, int(round(radius_m / MAP_RES)))
    near_paint = cv2.dilate(lane, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    observed = sem_map_bgr.any(axis=2)
    e = to_ego(np.asarray(pts, float))
    seg = np.linalg.norm(np.diff(e, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], max(2, int(s[-1] / step_m) + 1))
    q = np.stack([np.interp(t, s, e[:, 0]), np.interp(t, s, e[:, 1])], 1)
    u = np.round((MAP_Y - q[:, 1]) / MAP_RES).astype(int)
    v = np.round((MAP_X - q[:, 0]) / MAP_RES).astype(int)
    inside = (u >= 0) & (u < lane.shape[1]) & (v >= 0) & (v < lane.shape[0])
    if not inside.any():
        return 0.0, 0.0
    obs = observed[v[inside], u[inside]]
    if not obs.any():
        return 0.0, 0.0
    return float(obs.sum() / len(q)), float(near_paint[v[inside], u[inside]][obs].mean())


def to_ego(pts):
    """LiDAR-frame (x right, y forward) -> ego (x forward, y left)."""
    return np.stack([pts[..., 1] + LIDAR_X_IN_EGO, -pts[..., 0]], -1)


def ego_to_px(p, scale):
    """Ego (x fwd, y left) metres -> pixel (u, v) in the upscaled 300x150 map."""
    u = (MAP_Y - p[:, 1]) / MAP_RES * scale
    v = (MAP_X - p[:, 0]) / MAP_RES * scale
    return np.stack([u, v], 1)


def draw_frame(sem_map_bgr, front_bgr, vectors, scale=3):
    m = cv2.resize(sem_map_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    m = (0.55 * m).astype(np.uint8)
    for v in vectors:
        color = CLASS_BGR[MAP_CLASSES.index(v['cls'])]
        px = ego_to_px(np.array(v['pts']), scale).round().astype(np.int32)
        cv2.polylines(m, [px], False, color, 2, cv2.LINE_AA)
        cv2.circle(m, tuple(int(c) for c in px[-1]), 3, color, -1)
    c = ego_to_px(np.zeros((1, 2)), scale)[0].round().astype(int)  # ego marker (rear axle)
    cv2.rectangle(m, (c[0] - 5, c[1] - 11), (c[0] + 5, c[1] + 4), (255, 255, 255), 1)
    for i, name in enumerate(MAP_CLASSES):
        cv2.putText(m, name, (5, 15 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLASS_BGR[i], 1, cv2.LINE_AA)
    if front_bgr is None:
        return m
    h = m.shape[0]
    f = cv2.resize(front_bgr, (int(front_bgr.shape[1] * h / front_bgr.shape[0]), h))
    return np.hstack([f, m])


def draw_global(frames, all_vectors, path, res=0.2, margin=40.0):
    poses = np.array([f['pose'] for f in frames])
    world = []
    for f, vecs in zip(frames, all_vectors):
        x, y, yaw = f['pose']
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        for v in vecs:
            world.append((v['cls'], np.array(v['pts']) @ R.T + [x, y]))
    pts = np.concatenate([poses[:, :2]] + [w for _, w in world])
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    W, H = (hi - lo) / res
    scale = min(1.0, 6000 / max(W, H))
    W, H = max(int(W * scale), 1), max(int(H * scale), 1)

    def to_px(p):
        return np.stack([(p[:, 0] - lo[0]) / res * scale, H - (p[:, 1] - lo[1]) / res * scale], 1).round().astype(np.int32)

    canvas = np.full((H, W, 3), 25, np.uint8)
    for cls, w in world:
        cv2.polylines(canvas, [to_px(w)], False, CLASS_BGR[MAP_CLASSES.index(cls)], 1, cv2.LINE_AA)
    cv2.polylines(canvas, [to_px(poses[:, :2])], False, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='output dir of mcap_semantic_mapping.py')
    ap.add_argument('--config', default='projects/configs/semvecnet/semvecnet_nusc_centerline.py')
    ap.add_argument('--checkpoint', default='ckpts/mAP_epoch_99.pth')
    ap.add_argument('--score-thr', type=float, default=0.4)
    ap.add_argument('--class-thr', default='', help='per-class overrides of --score-thr for vectors/ and viz/, '
                    'e.g. divider=0.1 (predictions.json always keeps every prediction)')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--divider-paint-check', action='store_true',
                    help='drop predicted dividers not backed by lane paint in the semantic map')
    ap.add_argument('--paint-radius', type=float, default=0.6, help='m from paint that still counts as support')
    ap.add_argument('--paint-support', type=float, default=0.15, help='min fraction of the observed line near paint '
                    '(invented dividers score ~0; real ones with sparse paint ahead ~0.2)')
    ap.add_argument('--paint-observed', type=float, default=0.3, help='min fraction of the line over mapped cells')
    args = ap.parse_args()

    run = Path(args.run)
    frames = [json.loads(l) for l in (run / 'frames.jsonl').read_text().splitlines() if l.strip()]
    if not frames:
        sys.exit(f'No frames in {run / "frames.jsonl"}')
    (run / 'vectors').mkdir(exist_ok=True)
    (run / 'viz').mkdir(exist_ok=True)

    meta = json.loads((run / 'meta.json').read_text()) if (run / 'meta.json').exists() else {}
    ego_frame = meta.get('ego_frame') or 'base_link'
    model, norm = build_model(args.config, args.checkpoint, args.device)
    thr = {c: args.score_thr for c in MAP_CLASSES}
    thr.update({k: float(v) for k, v in (kv.split('=') for kv in args.class_thr.split(',') if kv)})
    print(f'Loaded {args.checkpoint}; {len(frames)} frames, thresholds {thr}', flush=True)

    all_vectors, raw, raw_unfiltered, writer = [], [], [], None
    n_dropped = n_div = 0
    for b in range(0, len(frames), args.batch):
        chunk = frames[b:b + args.batch]
        preds = predict(model, [load_map(run / f['map'], norm) for f in chunk], args.device)
        for f, (pts, scores, labels) in zip(chunk, preds):
            raw_unfiltered.append(dict(sample_token=str(f['idx']), vectors=[
                dict(pts=np.round(p, 4).tolist(), pts_num=len(p), cls_name=MAP_CLASSES[int(l)], type=int(l),
                     confidence_level=float(s)) for p, s, l in zip(pts, scores, labels)]))
            if args.divider_paint_check:
                sem = cv2.imread(str(run / f['map']))
                ok = np.ones(len(scores), bool)
                for i, (p, l) in enumerate(zip(pts, labels)):
                    if MAP_CLASSES[int(l)] == 'divider':
                        obs, sup = paint_support(p, sem, args.paint_radius)
                        ok[i] = obs >= args.paint_observed and sup >= args.paint_support
                        n_div += scores[i] >= 0.1
                        n_dropped += (not ok[i]) and scores[i] >= 0.1
                pts, scores, labels = pts[ok], scores[ok], labels[ok]
            raw.append(dict(sample_token=str(f['idx']), vectors=[
                dict(pts=np.round(p, 4).tolist(), pts_num=len(p), cls_name=MAP_CLASSES[int(l)], type=int(l),
                     confidence_level=float(s)) for p, s, l in zip(pts, scores, labels)]))
            keep = scores >= np.array([thr[MAP_CLASSES[int(l)]] for l in labels])
            vecs = [dict(cls=MAP_CLASSES[int(l)], score=round(float(s), 3), pts=np.round(to_ego(p), 3).tolist())
                    for p, s, l in zip(pts[keep], scores[keep], labels[keep])]
            all_vectors.append(vecs)
            (run / 'vectors' / f'{f["idx"]:06d}.json').write_text(json.dumps(dict(
                stamp_ns=f['stamp_ns'], pose=f['pose'], frame=f'{ego_frame} (x fwd, y left, m)', vectors=vecs)))
            front = cv2.imread(str(run / f['front'])) if f.get('front') else None
            img = draw_frame(cv2.imread(str(run / f['map'])), front, vecs)
            cv2.imwrite(str(run / 'viz' / f'{f["idx"]:06d}.png'), img)
            if not args.no_video:
                if writer is None:
                    writer = cv2.VideoWriter(str(run / 'semvecnet.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                             float(f.get('fps', 2.0)), (img.shape[1], img.shape[0]))
                writer.write(img)
        print(f'  {min(b + args.batch, len(frames))}/{len(frames)} frames', flush=True)
    if writer is not None:
        writer.release()
    (run / 'predictions.json').write_text(json.dumps(dict(
        meta=dict(frame='SemVecNet/nuScenes-LiDAR (x right, y forward, m)', lidar_x_in_ego=LIDAR_X_IN_EGO,
                  checkpoint=args.checkpoint),
        results=raw)))
    if args.divider_paint_check:
        (run / 'predictions_unfiltered.json').write_text(json.dumps(dict(results=raw_unfiltered)))
        print(f'Paint check: dropped {n_dropped} of {n_div} dividers (score >= 0.1) with no lane paint under them', flush=True)

    draw_global(frames, all_vectors, run / 'global_vectors.png')
    by_cls = {c: sum(v['cls'] == c for vs in all_vectors for v in vs) for c in MAP_CLASSES}
    print(f'Done: {sum(by_cls.values())} vectors over {len(frames)} frames {by_cls}')


if __name__ == '__main__':
    main()
