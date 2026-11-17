#!/usr/bin/env python3
"""Experiment: single-pass full-frame (100x100 m) SemVecNet on BEVFusion maps, zero-shot.

variant 'native': 200x200 input @0.5 m/px, bev 200x200
variant 'up':     nearest-upsampled to 500x500 @0.2 m/px (scale the CNN was trained on), bev 333x333
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/opt')
import infer_bevfusion as ib  # noqa: E402


def build(bev, half=50.0):
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    import importlib
    cfg = Config.fromfile('projects/configs/semvecnet/semvecnet_nusc_centerline.py')
    sys.path.insert(0, str(Path.cwd()))
    importlib.import_module(cfg.plugin_dir.strip('/').replace('/', '.'))
    from mmdet3d.models import build_model
    pc = [-half, -half, -10.0, half, half, 10.0]
    h = cfg.model.pts_bbox_head
    h.bev_h = h.bev_w = bev
    h.bbox_coder.pc_range = pc
    h.bbox_coder.post_center_range = [-60, -60, -60, -60, 60, 60, 60, 60]
    h.transformer.encoder.pc_range = pc
    h.positional_encoding.row_num_embed = h.positional_encoding.col_num_embed = bev
    cfg.model.pretrained = dict(sem=None)
    cfg.model.train_cfg = None
    m = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(m, 'ckpts/mAP_epoch_99.pth', map_location='cpu', strict=True)
    return m.cuda().eval()


def full_rgb(labels, res):
    """BEVFusion labels -> SemVecNet colours on a res m/px grid covering +-50 m, LiDAR-centred,
    image-left = ego-left (SemVecNet convention)."""
    n = int(round(2 * ib.BEVF_HALF / res))
    c = ib.BEVF_HALF - (np.arange(n) + 0.5) * res      # row: fwd, col: left
    rows = np.floor((ib.BEVF_HALF - c) / ib.BEVF_RES).astype(int)
    cols = np.floor((ib.BEVF_HALF + c) / ib.BEVF_RES).astype(int)
    return ib.BEVF_TO_SVN[labels[rows[:, None], cols[None, :]]]


def main():
    inp, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(inp.glob('*.png'))
    variants = {'native': (0.5, 200), 'up': (0.2, 333)}
    for name, (res, bev) in variants.items():
        model = build(bev)
        for f in files:
            labels = ib.decode_bevfusion(f)
            rgb = full_rgb(labels, res)
            with torch.no_grad():
                pts, scores, cls = ib.predict(model, [rgb], 'cuda:0')[0]
            vec = [(int(l), float(s), p) for p, s, l in zip(pts, scores, cls) if s >= 0.4]
            ib.visualize_full(out / f'{f.stem}_{name}.png', labels, vec,
                              [((0, 0), (-50, -50, 50, 50))], 0.4)
            print(name, f.name, {c: sum(v[0] == i for v in vec) for i, c in enumerate(ib.MAP_CLASSES)})
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
