# ./fine-tune.sh sets the mounted paths and manages evaluation and checkpoints.
_base_ = ['./semvecnet_nusc_centerline.py']

data_root = '/data/ringroad/'
dataset_type = 'RingRoadSemanticMapDataset'
load_from = 'ckpts/mAP_epoch_99.pth'

model = dict(pretrained=dict(sem=None))  # all weights come from load_from

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(type=dataset_type, augment=True, data_root=data_root, ann_file=data_root + 'train/train.pkl'),
    val=dict(type=dataset_type, data_root=data_root, ann_file=data_root + 'test/test.pkl',
             map_ann_file=data_root + 'test/test_gt.json'),
    test=dict(type=dataset_type, data_root=data_root, ann_file=data_root + 'test/test.pkl',
              map_ann_file=data_root + 'test/test_gt.json'))

optimizer = dict(lr=1e-4)               # nuScenes training started at 6e-4 from ImageNet weights
lr_config = dict(warmup_iters=100)
total_epochs = 60
runner = dict(max_epochs=total_epochs)
evaluation = dict(interval=1)           # chamfer mAP on test/ after every epoch; best kept (save_best inherited)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(interval=10)
