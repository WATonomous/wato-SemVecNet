"""Run the SemVecNet PyTorch checkpoint on a normalized semantic-map tensor."""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch

from export_semvecnet_onnx import Decoder, Encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Python SemVecNet directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="PyTorch .pth checkpoint")
    parser.add_argument("--input", type=Path, required=True, help="Normalized FP32 [1,3,300,150] .npy")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz file")
    args = parser.parse_args()

    sem_map = np.load(args.input, allow_pickle=False)
    if sem_map.shape != (1, 3, 300, 150) or sem_map.dtype != np.float32:
        raise ValueError("Expected an FP32 semantic map with shape [1, 3, 300, 150]")

    from mmcv import Config
    from mmcv.runner import load_checkpoint

    source = args.source.resolve()
    sys.path.insert(0, str(source))
    config = Config.fromfile(str(source / "projects/configs/semvecnet/semvecnet_nusc_centerline.py"))
    importlib.import_module(config.plugin_dir.strip("/").replace("/", "."))
    from mmdet3d.models import build_model

    config.model.pretrained = dict(sem=None)
    config.model.train_cfg = None
    model = build_model(config.model, test_cfg=config.get("test_cfg")).cpu().eval()
    load_checkpoint(model, str(args.checkpoint), map_location="cpu", strict=True)
    encoder, decoder = Encoder(model).eval(), Decoder(model.pts_bbox_head).eval()

    with torch.no_grad():
        features = encoder(torch.from_numpy(sem_map))
        logits, points = decoder(features)

    np.savez(args.output, class_logits=logits.cpu().numpy(), normalized_points=points.cpu().numpy())


if __name__ == "__main__":
    main()
