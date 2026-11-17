#!/usr/bin/env python3
"""Run SemVecNet (pretrained, unmodified) on CUDA-BEVFusion segmentation maps.

BEVFusion map (input)                    SemVecNet training map (what the model expects)
---------------------------------------  ----------------------------------------------
200x200 px, 0.5 m/px, +-50 m             300x150 px, 0.2 m/px, x +-30 m (fwd), y +-15 m (left)
centred on the LiDAR                     centred on the ego (rear axle)
top = forward, image-right = ego-left    top = forward, image-left = ego-left
6 nuScenes classes, bg (240,240,240)     4 Mapillary classes, bg (0,0,0)
JPEG bytes (saved with .png extension)

Each SemVecNet pixel is sampled (nearest neighbour) from the BEVFusion map, so the
pretrained checkpoint is used as-is -- no retraining or weight surgery.

Modes
  tiles (default)  cover the full +-50 m map with 5x3 overlapping 60x30 m windows (one batched
                   forward pass; the middle one is the original ego crop), clip each window's
                   vectors to its core area, drop vectors not backed by segmented pixels
                   (the model hallucinates roads in empty windows), stitch seams, de-duplicate.
  crop             only the central 60x30 m window (the area the model was trained on).

Predicted vectors are in the LiDAR frame: x = right, y = forward, metres.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------- BEVFusion side
# Order matches CUDA-BEVFusion src/main.cpp visualize_segmap(); later classes overwrite earlier.
BEVF_COLORS = np.array([
    [240, 240, 240],  # 0 background
    [166, 206, 227],  # 1 drivable_area
    [31, 120, 180],   # 2 ped_crossing
    [178, 223, 138],  # 3 walkway
    [51, 160, 44],    # 4 stop_line
    [251, 154, 153],  # 5 carpark_area
    [227, 26, 28],    # 6 divider
], dtype=np.int32)
BEVF_RES, BEVF_HALF = 0.5, 50.0

# ---------------------------------------------------------------- SemVecNet side
# semantic_mapping/src/node_config/base_cfg.py LABEL_COLORS (RGB)
ROAD, CROSSWALK, LANE, SIDEWALK, BG = [128, 64, 128], [140, 140, 200], [255, 255, 255], [244, 35, 232], [0, 0, 0]
# bg->bg, drivable->road, ped_crossing->crosswalk, walkway->sidewalk,
# stop_line->road (BEVFusion predicts it as fat blobs, unlike thin paint), carpark->road, divider->lane
BEVF_TO_SVN = np.array([BG, ROAD, CROSSWALK, SIDEWALK, ROAD, ROAD, LANE], dtype=np.uint8)
SVN_RES, SVN_X, SVN_Y = 0.2, 30.0, 15.0           # base_cfg MAPPING.RESOLUTION / nuscenes_cfg MAPPING.RANGE
IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
LIDAR_X_IN_EGO = 0.943713                          # nuScenes LIDAR_TOP -> ego translation (forward, m)

MAP_CLASSES = ['divider', 'ped_crossing', 'boundary', 'centerline']
CLASS_COLORS = ['orange', 'blue', 'red', 'green']
PED, CENTERLINE = 1, 3

# ---------------------------------------------------------------- tiling (LiDAR frame)
# Window centres (60x30 m windows). The middle window is exactly the original ego-centred crop:
# the model has an ego-centric prior, so the real road must not sit on a seam.
WIN_CY = np.array([-25.0, 0.0, 25.0])              # 35 m forward overlap, covers +-55 m
WIN_CX = np.linspace(-BEVF_HALF + SVN_Y, BEVF_HALF - SVN_Y, 5)  # -35..35 step 17.5, 12.5 m overlap
MAX_SUPPORT_DIST = 2.0  # drop vectors whose points are on average further (m) from any segmented pixel


def decode_bevfusion(path):
    """Colour PNG/JPEG -> (200,200) class-id map, robust to JPEG artefacts."""
    rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.int32)
    d = ((rgb[:, :, None, :] - BEVF_COLORS[None, None]) ** 2).sum(-1)
    return d.argmin(-1).astype(np.uint8)


def to_semvecnet_rgb(labels, lidar_x_in_ego=LIDAR_X_IN_EGO, center=(0.0, 0.0)):
    """Resample a BEVFusion class map onto SemVecNet's 300x150 ego grid and colourise.

    center = (x, y) of the virtual LiDAR in the BEVFusion LiDAR frame (tile offset)."""
    h, w = int(2 * SVN_X / SVN_RES), int(2 * SVN_Y / SVN_RES)
    ego_fwd = SVN_X - (np.arange(h) + 0.5) * SVN_RES     # row 0 = +30 m (forward)
    ego_left = SVN_Y - (np.arange(w) + 0.5) * SVN_RES    # col 0 = +15 m (left)
    fwd = ego_fwd - lidar_x_in_ego + center[1]           # BEVFusion grid is LiDAR-centred
    left = ego_left - center[0]                          # LiDAR x = right = -left
    rows = np.floor((BEVF_HALF - fwd) / BEVF_RES).astype(int)
    cols = np.floor((BEVF_HALF + left) / BEVF_RES).astype(int)  # BEVFusion image is mirrored
    H, W = labels.shape
    ok = ((rows >= 0) & (rows < H))[:, None] & ((cols >= 0) & (cols < W))[None, :]
    lab = labels[np.clip(rows, 0, H - 1)[:, None], np.clip(cols, 0, W - 1)[None, :]]
    return BEVF_TO_SVN[np.where(ok, lab, 0)]


def make_tiles(lidar_x_in_ego, mode):
    """-> list of (center_xy, core_box) ; core boxes partition the covered area."""
    if mode == 'crop':
        y0 = -SVN_X - lidar_x_in_ego
        return [((0.0, 0.0), (-SVN_Y, y0, SVN_Y, y0 + 2 * SVN_X))]
    xb = np.r_[-BEVF_HALF, (WIN_CX[1:] + WIN_CX[:-1]) / 2, BEVF_HALF]
    yb = np.r_[-BEVF_HALF, (WIN_CY[1:] + WIN_CY[:-1]) / 2, BEVF_HALF]
    tiles = []
    for j, wy in enumerate(WIN_CY):
        for i, wx in enumerate(WIN_CX):
            # window centre wy corresponds to virtual LiDAR at wy + lidar_x_in_ego
            # (middle window: virtual LiDAR at 0, identical to crop mode)
            cy = 0.0 if wy == 0 else float(wy + lidar_x_in_ego)
            tiles.append(((float(wx), cy), (xb[i], yb[j], xb[i + 1], yb[j + 1])))
    return tiles


def support_fn(labels):
    """-> f(pts[N,2]) = mean distance (m) from each point to the nearest segmented BEVFusion pixel."""
    from scipy.ndimage import distance_transform_edt
    content = labels[:, ::-1] != 0          # un-mirrored: col increases with x (right)
    dist = distance_transform_edt(~content) * BEVF_RES
    H, W = dist.shape

    def f(p):
        c = np.floor((p[:, 0] + BEVF_HALF) / BEVF_RES).astype(int)
        r = np.floor((BEVF_HALF - p[:, 1]) / BEVF_RES).astype(int)
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        d = np.full(len(p), 1e3)
        d[ok] = dist[r[ok], c[ok]]
        return d.mean()
    return f


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
    return model.to(device).eval()


@torch.no_grad()
def predict(model, rgbs, device):
    """rgbs: list of (300,150,3) uint8 -> list of (pts[N,20,2], scores[N], labels[N])."""
    x = np.stack([(r.astype(np.float32) - IMG_MEAN) / IMG_STD for r in rgbs])
    x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)
    meta = [dict(scene_token='bevfusion', can_bus=np.zeros(18), pad_shape=[r.shape], img_shape=[r.shape])
            for r in rgbs]
    _, results = model.simple_test(meta, sem_map=x)
    return [(r['pts_bbox']['pts_3d'].numpy(), r['pts_bbox']['scores_3d'].numpy(),
             r['pts_bbox']['labels_3d'].numpy()) for r in results]


# ---------------------------------------------------------------- merging
def _resample(coords, n=20):
    from shapely.geometry import LineString
    line = LineString(coords)
    return np.array([line.interpolate(t, normalized=True).coords[0] for t in np.linspace(0, 1, n)])


def _pieces(geom, min_len):
    """Flatten a shapely intersection result into coordinate arrays."""
    if geom.is_empty:
        return []
    if geom.geom_type == 'LineString':
        return [np.array(geom.coords)] if geom.length >= min_len else []
    if geom.geom_type == 'Polygon':
        return [np.array(geom.exterior.coords)] if geom.area >= 1.0 else []
    if hasattr(geom, 'geoms'):
        return [p for g in geom.geoms for p in _pieces(g, min_len)]
    return []


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _stitch(lines, directed, gap=1.5, min_cos=0.7):
    """Greedily join same-class polylines whose ends meet. lines: list of [coords, score]."""
    merged = True
    while merged:
        merged = False
        for a in range(len(lines)):
            for b in range(len(lines)):
                if a == b:
                    continue
                pa, pb = lines[a][0], lines[b][0]
                options = [(pa, pb)] if directed else [(pa, pb), (pa, pb[::-1]), (pa[::-1], pb)]
                for qa, qb in options:
                    if np.linalg.norm(qa[-1] - qb[0]) < gap and \
                            _unit(qa[-1] - qa[-2]) @ _unit(qb[1] - qb[0]) > min_cos:
                        lines[a] = [np.vstack([qa, qb[1:]]), max(lines[a][1], lines[b][1])]
                        del lines[b]
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break
    return lines


def merge_tiles(tile_preds, tiles, thr, support, min_len=1.0, dedupe=1.0):
    """Clip each tile's vectors to its core box, drop unsupported ones, stitch seams, de-duplicate."""
    from shapely.geometry import LineString, Polygon, box
    from shapely.ops import unary_union
    by_cls = {c: [] for c in range(len(MAP_CLASSES))}
    for ((cx, cy), core), (pts, scores, labels) in zip(tiles, tile_preds):
        core = box(*core)
        for p, s, l in zip(pts, scores, labels):
            if s < thr:
                continue
            p = p + np.array([cx, cy])
            if int(l) == PED:
                g = Polygon(p).buffer(0)
            else:
                g = LineString(p)
            for piece in _pieces(g.intersection(core), min_len):
                if support(_resample(piece)) <= MAX_SUPPORT_DIST:
                    by_cls[int(l)].append([piece, float(s)])

    out = []
    for c, items in by_cls.items():
        if not items:
            continue
        if c == PED:  # union crossing pieces that touch across a seam
            blobs = unary_union([Polygon(p).buffer(0.5) for p, _ in items])
            blobs = list(blobs.geoms) if hasattr(blobs, 'geoms') else [blobs]
            for b in blobs:
                b = b.buffer(-0.5)
                if b.is_empty:
                    continue
                score = max(s for p, s in items if Polygon(p).buffer(0.5).intersects(b.buffer(0.5)))
                for piece in _pieces(b, min_len):
                    out.append((c, score, _resample(piece)))
            continue
        lines = _stitch(items, directed=(c == CENTERLINE))
        res = sorted([(s, _resample(p)) for p, s in lines], key=lambda t: -t[0])
        kept = []
        for s, p in res:  # chamfer de-dup, keep higher score
            if all(np.mean(np.min(np.linalg.norm(p[:, None] - q[None], axis=-1), axis=1)) > dedupe
                   for _, q in kept):
                kept.append((s, p))
        out += [(c, s, p) for s, p in kept]
    return out  # list of (class, score, pts[20,2])


# ---------------------------------------------------------------- visualisation
def _draw_vectors(ax, vectors):
    for l, s, p in vectors:
        c = CLASS_COLORS[l]
        ax.plot(p[:, 0], p[:, 1], color=c, lw=2)
        ax.scatter(p[:, 0], p[:, 1], color=c, s=4)
        if l == CENTERLINE:
            ax.annotate('', xy=p[-1], xytext=p[-2], arrowprops=dict(arrowstyle='->', color=c, lw=2))
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-0.9, -2.0), 1.8, 4.5, color='k'))  # ego car
    for i, name in enumerate(MAP_CLASSES):
        ax.plot([], [], color=CLASS_COLORS[i], label=name)
    ax.legend(loc='upper right', fontsize=8)


def visualize_crop(path, bevf_rgb, svn_rgb, vectors, thr, lidar_x_in_ego):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 9), gridspec_kw=dict(width_ratios=[2, 1, 1]))
    ax[0].imshow(bevf_rgb)
    # SemVecNet crop, drawn in BEVFusion pixel coords (image-right = ego-left)
    f0, f1 = -SVN_X - lidar_x_in_ego, SVN_X - lidar_x_in_ego
    r0, r1 = (BEVF_HALF - f1) / BEVF_RES, (BEVF_HALF - f0) / BEVF_RES
    c0, c1 = (BEVF_HALF - SVN_Y) / BEVF_RES, (BEVF_HALF + SVN_Y) / BEVF_RES
    ax[0].add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec='k', lw=1.5, ls='--'))
    ax[0].set_title('BEVFusion output (crop used = dashed)')
    ext = [-SVN_Y, SVN_Y, -SVN_X - lidar_x_in_ego, SVN_X - lidar_x_in_ego]
    ax[1].imshow(svn_rgb, extent=ext)
    ax[1].set_title('SemVecNet input (300x150)')
    ax[2].imshow(svn_rgb, extent=ext, alpha=0.35)
    _draw_vectors(ax[2], vectors)
    ax[2].set_xlim(-SVN_Y, SVN_Y)
    ax[2].set_ylim(-SVN_X, SVN_X)
    ax[2].set_title(f'SemVecNet vectors ({len(vectors)} with score >= {thr})')
    for a in ax:
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def visualize_full(path, labels, vectors, tiles, thr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ext = [-BEVF_HALF, BEVF_HALF, -BEVF_HALF, BEVF_HALF]
    lab = labels[:, ::-1]  # un-mirror: x (right) increases left->right
    fig, ax = plt.subplots(1, 2, figsize=(18, 9.5))
    ax[0].imshow(BEVF_COLORS[lab].astype(np.uint8), extent=ext)
    for _, (x0, y0, x1, y1) in tiles:
        ax[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='k', lw=1, ls='--'))
    ax[0].set_title('BEVFusion output (un-mirrored; dashed = tile cores)')
    ax[1].imshow(BEVF_TO_SVN[lab], extent=ext, alpha=0.35)
    _draw_vectors(ax[1], vectors)
    ax[1].set_xlim(-BEVF_HALF, BEVF_HALF)
    ax[1].set_ylim(-BEVF_HALF, BEVF_HALF)
    ax[1].set_title(f'SemVecNet vectors, {len(tiles)} tiles merged ({len(vectors)} with score >= {thr})')
    for a in ax:
        a.set_xlabel('x right (m)'), a.set_ylabel('y forward (m)')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='BEVFusion seg PNG or a directory of them')
    ap.add_argument('--output', required=True, help='output directory')
    ap.add_argument('--mode', choices=['tiles', 'crop'], default='tiles')
    ap.add_argument('--config', default='projects/configs/semvecnet/semvecnet_nusc_centerline.py')
    ap.add_argument('--checkpoint', default='ckpts/mAP_epoch_99.pth')
    ap.add_argument('--score-thr', type=float, default=0.4, help='visualization / JSON score threshold')
    ap.add_argument('--lidar-x-in-ego', type=float, default=LIDAR_X_IN_EGO)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    inp, out = Path(args.input), Path(args.output)
    files = sorted(inp.glob('*.png')) + sorted(inp.glob('*.jpg')) if inp.is_dir() else [inp]
    if not files:
        sys.exit(f'No .png/.jpg files found in {inp}')
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(args.config, args.checkpoint, args.device)
    tiles = make_tiles(args.lidar_x_in_ego, args.mode)
    print(f'Loaded {args.checkpoint}; mode={args.mode} ({len(tiles)} window(s)); running on {len(files)} map(s)')

    summary = {}
    for f in files:
        labels = decode_bevfusion(f)
        inputs = [to_semvecnet_rgb(labels, args.lidar_x_in_ego, c) for c, _ in tiles]
        preds = predict(model, inputs, args.device)
        if args.mode == 'crop':
            pts, scores, cls = preds[0]
            vectors = [(int(l), float(s), p) for p, s, l in zip(pts, scores, cls) if s >= args.score_thr]
            Image.fromarray(inputs[0]).save(out / f'{f.stem}_semvecnet_input.png')
            visualize_crop(out / f'{f.stem}_vectors.png', BEVF_COLORS[labels].astype(np.uint8), inputs[0],
                           vectors, args.score_thr, args.lidar_x_in_ego)
        else:
            vectors = merge_tiles(preds, tiles, args.score_thr, support_fn(labels))
            visualize_full(out / f'{f.stem}_vectors.png', labels, vectors, tiles, args.score_thr)
        vecs = [dict(label=MAP_CLASSES[l], score=round(s, 4), pts=np.round(p, 3).tolist()) for l, s, p in vectors]
        with open(out / f'{f.stem}_vectors.json', 'w') as fh:
            json.dump(dict(frame='lidar (x right, y forward), metres', mode=args.mode,
                           tiles=[dict(center=list(c), core=list(map(float, b))) for c, b in tiles],
                           vectors=vecs), fh)
        counts = {c: sum(v['label'] == c for v in vecs) for c in MAP_CLASSES}
        summary[f.name] = counts
        print(f'  {f.name}: ' + ', '.join(f'{k}={v}' for k, v in counts.items()))

    with open(out / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'Done -> {out}')


if __name__ == '__main__':
    main()
