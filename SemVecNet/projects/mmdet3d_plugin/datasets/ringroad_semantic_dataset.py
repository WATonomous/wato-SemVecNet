"""Ring-road semantic-map dataset (tools/build_finetune_dataset.py output) for fine-tuning SemVecNet.

Each info of <split>.pkl: token, scene_token (run), frame_idx, timestamp, sem_path (relative to the pkl's directory),
ego_xy / yaw (offset-corrected UTM pose), annotation = {class: [N x 2 float32 polylines]} in the model frame
(x right, y forward, origin 0.944 m ahead of the rear axle) - the same frame and GT construction as nuScenes MapTRv2.
Everything else (GT sampling, aux BEV segmentation targets, chamfer evaluation) is the parent class unchanged.
"""
import os.path as osp

import numpy as np
from mmdet.datasets import DATASETS
from nuscenes.eval.common.utils import Quaternion

from .ringroad_augmentation import RingRoadAffine
from .nuscenes_offlinemap_dataset import CustomNuScenesOfflineLocalSemanticMapDataset


@DATASETS.register_module()
class RingRoadSemanticMapDataset(CustomNuScenesOfflineLocalSemanticMapDataset):

    def __init__(self, pipeline, augment=False, **kwargs):
        if augment and not kwargs.get('test_mode', False):
            pipeline = list(pipeline)
            normalize = next(i for i, step in enumerate(pipeline) if step['type'] == 'NormalizeSemanticMap')
            pipeline.insert(normalize, RingRoadAffine())
        super().__init__(pipeline=pipeline, **kwargs)

    def prepare_test_data(self, index):
        example = super().prepare_test_data(index)
        # MapTRv2.forward_test unwraps the augmentation dimension on metadata
        # while accepting the batched semantic tensor directly.
        example['img_metas'] = [example['img_metas']]
        return example

    def get_data_info(self, index):
        info = self.data_infos[index]
        x, y = (float(v) for v in info['ego_xy'])
        yaw = float(info['yaw'])
        q = Quaternion(axis=[0, 0, 1], radians=yaw)
        ego2global = np.eye(4)
        ego2global[:3, :3] = q.rotation_matrix
        ego2global[:2, 3] = [x, y]
        can_bus = np.zeros(18)
        can_bus[:3] = [x, y, 0.0]
        can_bus[3:7] = q.elements
        deg = np.degrees(yaw) % 360
        can_bus[-2], can_bus[-1] = deg / 180 * np.pi, deg
        return dict(
            sample_idx=info['token'], scene_token=info['scene_token'], frame_idx=info['frame_idx'],
            timestamp=info['timestamp'], prev_idx='', next_idx='',
            sem_filename=osp.join(osp.dirname(osp.abspath(self.ann_file)), info['sem_path']),
            ann_info=info['annotation'], can_bus=can_bus,
            ego2global_translation=[x, y, 0.0], ego2global_rotation=list(q.elements),
            lidar2ego=np.eye(4), lidar2global=ego2global, map_location='ring_road')
