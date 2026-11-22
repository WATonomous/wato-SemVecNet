"""Single-GPU Ring Road fine-tuning. Run inside the SemVecNet image."""
import argparse
import copy
import csv
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import HOOKS, Hook, save_checkpoint
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.apis import set_random_seed, single_gpu_test
from mmdet.core import EvalHook
from mmdet.utils import get_root_logger
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    if min(args.batch, args.epochs) < 1 or args.workers < 0 or not 0 < args.lr < math.inf:
        parser.error('epochs, batch and lr must be positive and workers nonnegative')
    return args


def evaluate(model, loader):
    predictions = single_gpu_test(model, loader)
    return loader.dataset.evaluate(predictions, metric='chamfer')


@HOOKS.register_module()
class RingRoadMetrics(Hook):
    def __init__(self, output, train_eval_cfg, workers, interval):
        from projects.mmdet3d_plugin.datasets.builder import build_dataloader

        self.output = Path(output)
        subset = build_dataset(train_eval_cfg)
        indices = np.random.RandomState(0).choice(len(subset), min(150, len(subset)), replace=False)
        subset.data_infos = [subset.data_infos[i] for i in sorted(indices)]
        self.train_eval = build_dataloader(subset, 1, workers, dist=False, shuffle=False)
        self.interval = interval
        self.rows = []
        self.best = -math.inf
        self.best_epoch = 0

    def checkpoint(self, runner, name, epoch):
        target = self.output / name
        tmp = target.with_suffix('.tmp')
        save_checkpoint(runner.model.module, str(tmp), meta=dict(epoch=epoch, CLASSES=runner.model.module.CLASSES))
        os.replace(tmp, target)

    def record(self, runner, epoch, loss, started, test):
        print('[EVAL] epoch {} fixed train subset'.format(epoch), flush=True)
        train = evaluate(runner.model, self.train_eval)
        prefix = 'NuscMap_chamfer/'
        score = float(test[prefix + 'mAP'])
        row = dict(epoch=epoch, mean_loss=loss, train_mAP=float(train[prefix + 'mAP']), test_mAP=score)
        for name in self.train_eval.dataset.MAPCLASSES:
            row[name + '_AP'] = float(test[prefix + name + '_AP'])
        if not all(math.isfinite(v) for v in row.values() if v is not None):
            raise ValueError('Nonfinite epoch metrics: {}'.format(row))
        if score > self.best:
            self.checkpoint(runner, 'best.pth', epoch)
            self.best, self.best_epoch = score, epoch
        if epoch:
            self.checkpoint(runner, 'latest.pth', epoch)
        elapsed = time.monotonic() - started
        self.rows.append(row)
        row.update(best_test_mAP=self.best, best_epoch=self.best_epoch, seconds=round(elapsed, 1),
                   peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        epoch_times = [r['seconds'] for r in self.rows if r['epoch']]
        row['eta_seconds'] = round(np.mean(epoch_times) * (runner.max_epochs - epoch)) if epoch_times else None
        for ext in ('json', 'csv'):
            path = self.output / ('metrics.' + ext)
            tmp = path.with_suffix('.' + ext + '.tmp')
            with tmp.open('w', newline='') as stream:
                if ext == 'json':
                    json.dump(self.rows, stream, indent=2, allow_nan=False)
                else:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(self.rows)
            os.replace(tmp, path)
        print('[SUMMARY] ' + ' '.join('{}={:.5f}'.format(k, v) if isinstance(v, float)
                                      else '{}={}'.format(k, v) for k, v in row.items()), flush=True)

    def before_run(self, runner):
        eval_hook = next(hook for hook in runner.hooks if isinstance(hook, EvalHook))
        # Override the plugin's timestamped prediction directory. NuScenesDataset
        # creates and cleans a TemporaryDirectory when the prefix is None.
        eval_hook.eval_kwargs['jsonfile_prefix'] = None
        test_loader = eval_hook.dataloader
        started = time.monotonic()
        print('[EVAL] baseline on {} test maps'.format(len(test_loader.dataset)), flush=True)
        self.record(runner, 0, None, started, evaluate(runner.model, test_loader))

    def before_train_epoch(self, runner):
        self.started = time.monotonic()
        self.total = 0.
        self.samples = 0

    def after_train_iter(self, runner):
        logs = runner.outputs['log_vars']
        loss = float(logs['loss'])
        if not math.isfinite(loss):
            raise FloatingPointError('Nonfinite loss at iteration {}'.format(runner.iter + 1))
        n = runner.outputs['num_samples']
        self.total += loss * n
        self.samples += n
        i, length = runner.inner_iter + 1, len(runner.data_loader)
        if i == 1 or i % self.interval == 0 or i == length:
            terms = ' '.join('{}={:.4f}'.format(k, v) for k, v in logs.items()
                             if k in ('loss_cls', 'loss_pts', 'loss_dir', 'loss_seg'))
            eta = (time.monotonic() - self.started) / i * (length - i)
            print('[TRAIN] epoch {}/{} iter {}/{} loss {:.5f} ({}) lr {:.3g} epoch_eta {:.0f}s'.format(
                runner.epoch + 1, runner.max_epochs, i, length, loss, terms, max(runner.current_lr()), eta),
                flush=True)

    def after_train_epoch(self, runner):
        # The standard EvalHook (LOW priority) has already populated the test metrics.
        self.record(runner, runner.epoch + 1, self.total / self.samples, self.started, runner.log_buffer.output)
        runner.log_buffer.clear()

    def after_run(self, runner):
        final = self.output / 'ringroad_finetuned.pth'
        from mcap_infer_semvecnet import build_model as build_inference_model

        best = self.output / 'best.pth'
        # Use the production strict loader on CPU before publishing the export.
        build_inference_model('projects/configs/semvecnet/semvecnet_nusc_centerline.py', str(best), 'cpu')
        os.link(best, final)
        print('Selected {} (epoch {}), baseline test mAP {:.5f}, best {:.5f}'.format(
            final, self.best_epoch, self.rows[0]['test_mAP'], self.best), flush=True)


def main():
    args = parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Use a fresh output directory to avoid mixing checkpoints or cached ground truth: ' + str(out))
    cfg = Config.fromfile('projects/configs/semvecnet/semvecnet_ringroad_finetune.py')
    sys.path.insert(0, str(Path.cwd()))
    importlib.import_module(cfg.plugin_dir.strip('/').replace('/', '.'))
    from projects.mmdet3d_plugin.bevformer.apis.train import custom_train_model

    cfg.seed = 0
    set_random_seed(cfg.seed)
    cfg.data_root = str(args.data_root.resolve())
    for name, split in [('train', 'train'), ('val', 'test'), ('test', 'test')]:
        data = cfg.data[name]
        data.data_root = cfg.data_root
        data.ann_file = str(Path(cfg.data_root) / split / (split + '.pkl'))
        data.map_ann_file = str(out / (split + '_gt.json'))
    cfg.data.train.augment = bool(args.augment)
    cfg.data.samples_per_gpu, cfg.data.workers_per_gpu = args.batch, args.workers
    cfg.total_epochs = cfg.runner.max_epochs = args.epochs
    cfg.optimizer.lr = args.lr
    cfg.load_from = args.checkpoint
    train = build_dataset(cfg.data.train)
    train_eval_cfg = copy.deepcopy(cfg.data.test)
    train_eval_cfg.ann_file = cfg.data.train.ann_file
    train_eval_cfg.map_ann_file = cfg.data.train.map_ann_file
    cfg.work_dir = str(out)
    cfg.gpu_ids = [0]
    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer.type == 'AdamW':
        cfg.optimizer.type = 'AdamW2'
    cfg.evaluation.save_best = None
    cfg.checkpoint_config = None
    cfg.custom_hooks = [dict(type='RingRoadMetrics', output=str(out), train_eval_cfg=train_eval_cfg,
                             workers=args.workers, interval=cfg.log_config.interval, priority='VERY_LOW')]
    cfg.log_config = None
    # Initialize the shared mmdet logger before the plugin looks it up.
    get_root_logger(log_level=cfg.log_level)
    print('GPU={} train={} train_eval={} iters/epoch={} epochs={} batch={} workers={} '
          'lr={} fp16={} augment={}'.format(torch.cuda.get_device_name(0), len(train), min(150, len(train)),
          math.ceil(len(train) / args.batch), args.epochs, args.batch, args.workers, args.lr,
          cfg.fp16, cfg.data.train.augment), flush=True)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    # MapTR installs a Pretrained initializer even when the backbone path is None.
    # The runner loads the full starting checkpoint after the normal model initialization.
    model.sem_map_backbone.init_cfg = None
    model.init_weights()
    model.CLASSES = train.CLASSES
    custom_train_model(model, [train], cfg, distributed=False, validate=True, meta=dict(seed=cfg.seed))


if __name__ == '__main__':
    main()
