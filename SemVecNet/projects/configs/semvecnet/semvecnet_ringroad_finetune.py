# Fine-tune the nuScenes SemVecNet (mAP_epoch_99.pth) on ring-road semantic maps from tools/build_finetune_dataset.py.
# Model, input pipeline and loss are the nuScenes config unchanged; only the data, the schedule and the start
# weights differ. data_root is the dataset directory (train/train.pkl, test/test.pkl); test/ is the held-out drive.
#
#   python tools/train.py projects/configs/semvecnet/semvecnet_ringroad_finetune.py --work-dir work_dirs/ringroad
#
# Set data_root below to where the data/ folder is on the training machine (the ann_file paths are built from it
# when this file is loaded, so --cfg-options data_root=... would not reach them). load_from is relative to the
# SemVecNet directory. The test/ directory must be writable: the first evaluation writes test/test_gt.json there.
# 8 GB GPU: fp32 at samples_per_gpu=1 peaks at ~3.8 GB; fp16 (inherited from the base config) at 2 should fit.
_base_ = ['./semvecnet_nusc_centerline.py']

data_root = '/data/ringroad/'
dataset_type = 'RingRoadSemanticMapDataset'
load_from = 'ckpts/mAP_epoch_99.pth'

model = dict(pretrained=dict(sem=None))  # all weights come from load_from

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(type=dataset_type, data_root=data_root, ann_file=data_root + 'train/train.pkl'),
    val=dict(type=dataset_type, data_root=data_root, ann_file=data_root + 'test/test.pkl',
             map_ann_file=data_root + 'test/test_gt.json'),
    test=dict(type=dataset_type, data_root=data_root, ann_file=data_root + 'test/test.pkl',
              map_ann_file=data_root + 'test/test_gt.json'))

optimizer = dict(lr=1e-4)               # nuScenes training started at 6e-4 from ImageNet weights
lr_config = dict(warmup_iters=100)
total_epochs = 24
runner = dict(max_epochs=total_epochs)
evaluation = dict(interval=1)           # chamfer mAP on test/ after every epoch; best kept (save_best inherited)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(interval=10)
