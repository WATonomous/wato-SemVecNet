"""Export fixed-shape SemVecNet graphs using the repository's PyTorch image."""

import argparse
import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.sem_map_backbone
        self.neck = model.sem_map_neck

    def forward(self, sem_map):
        return self.neck(self.backbone(sem_map))[0]


class Decoder(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, features):
        out = self.head(None, None, features, img_metas=[{}])
        return out["all_cls_scores"][-1], out["all_pts_preds"][-1]


def sample_single_level(height, width, values, _shapes, locations, weights):
    """Export bilinear sampling with zero padding and align_corners=False."""
    batch, _, heads, channels = values.shape
    _, queries, _, _, samples, _ = locations.shape
    values = values.permute(0, 2, 1, 3).reshape(batch * heads, height * width, channels)
    locations = locations.squeeze(3).transpose(1, 2).reshape(batch * heads, -1, 2)
    x = locations[..., 0] * width - 0.5
    y = locations[..., 1] * height - 0.5
    x0, y0 = x.floor(), y.floor()
    dx, dy = x - x0, y - y0

    def pixel(ix, iy):
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        indices = ix.clamp(0, width - 1).long() + width * iy.clamp(0, height - 1).long()
        return values.gather(1, indices.unsqueeze(-1).expand(-1, -1, channels)) * valid.unsqueeze(-1)

    interpolated = (
        pixel(x0, y0) * ((1 - dx) * (1 - dy)).unsqueeze(-1)
        + pixel(x0 + 1, y0) * (dx * (1 - dy)).unsqueeze(-1)
        + pixel(x0, y0 + 1) * ((1 - dx) * dy).unsqueeze(-1)
        + pixel(x0 + 1, y0 + 1) * (dx * dy).unsqueeze(-1)
    )
    interpolated = interpolated.reshape(batch, heads, queries, samples, channels)
    weights = weights.squeeze(3).transpose(1, 2).unsqueeze(-1)
    return (interpolated * weights).sum(dim=3).transpose(1, 2).reshape(batch, queries, -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd(), help="Python SemVecNet directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Save PyTorch inputs and outputs for parity checks")
    args = parser.parse_args()

    from mmcv import Config
    from mmcv.runner import load_checkpoint

    sys.path.insert(0, str(args.source.resolve()))
    config = Config.fromfile(str(args.source / "projects/configs/semvecnet/semvecnet_nusc_centerline.py"))
    importlib.import_module(config.plugin_dir.strip("/").replace("/", "."))
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.bevformer.modules import decoder as attention_module

    config.model.pretrained = dict(sem=None)
    config.model.train_cfg = None
    model = build_model(config.model, test_cfg=config.get("test_cfg")).cpu().eval()
    load_checkpoint(model, str(args.source / "ckpts/mAP_epoch_99.pth"), map_location="cpu", strict=True)
    encoder, decoder = Encoder(model).eval(), Decoder(model.pts_bbox_head).eval()
    head = config.model.pts_bbox_head
    assert head.transformer.decoder.transformerlayers.attn_cfgs[-1]["num_levels"] == 1

    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    sem_map = torch.randn(1, 3, 300, 150)
    with torch.no_grad():
        features = encoder(sem_map)
        logits, points = decoder(features)
        torch.onnx.export(
            encoder, sem_map, str(args.output / "semvecnet_encoder.onnx"),
            input_names=["sem_map"], output_names=["features"], opset_version=13,
        )
        # Legacy PyTorch cannot export grid_sample. Keep the trained attention
        # projections and express only the one-level sampler with standard ops.
        def sample(values, shapes, locations, weights):
            return sample_single_level(head.bev_h, head.bev_w, values, shapes, locations, weights)

        with patch.object(attention_module, "multi_scale_deformable_attn_pytorch", sample):
            for actual, expected in zip(decoder(features), (logits, points)):
                torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            torch.onnx.export(
                decoder, features, str(args.output / "semvecnet_decoder.onnx"),
                input_names=["features"], output_names=["class_logits", "normalized_points"],
                opset_version=13,
            )
    if args.reference:
        np.savez(args.reference, sem_map=sem_map.numpy(), features=features.numpy(),
                 class_logits=logits.numpy(), normalized_points=points.numpy())


if __name__ == "__main__":
    main()
