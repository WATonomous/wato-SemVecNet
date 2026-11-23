# Ring Road Phase 1 on a GPU server

## WATcloud SLURM cluster

On a login node, from a fresh checkout:

```sh
./run_stage1_slurm.sh
```

The script installs rclone into `~/.local/bin` if it is missing and checks that the `gdrive` remote can see the Bags folder. Set up `gdrive` before the first run, either by copying a working `rclone.conf` into the repo root (`scp rclone.conf <server>:<repo>/`; it is gitignored because it holds your Google Drive token, and every script uses it ahead of `~/.config/rclone/rclone.conf`) or by running `rclone config` (see below). The script then submits itself with `sbatch` (8 CPUs, 32 GB, a 16 GiB GPU shard, 100 GiB tmpdisk, 1 day) and follows the job log at `logs/stage1-<job>.out`. Ctrl-C stops following the log; the job keeps running. Inside the job:

1. Starts rootless Docker (`slurm-start-dockerd.sh`), pulls `naruarjun/tensorrt-nvidia-semseg` into the job's `/tmp` and checks the GPU from inside a container.
2. Builds the HRNet TensorRT engine once per GPU model (~15 min), in `~/semvecnet-phase1/assets/hrnet_<gpu>.engine`.
3. For each split in `tools/phase1_plan_all.txt` (112 splits; `SEMVECNET_PLAN=tools/phase1_plan_b.txt` for the old eight): downloads it to `/tmp`, makes the semantic maps, aligns them to `ring_road.osm` and deletes the split. The log shows `[i/N]`, the split being downloaded (bytes / % / speed / ETA every 30 s), the run being processed, scan progress and the result.
4. Writes the dataset to `data/` in the repo: `train/` and `test/` (`maps/`, `labels/` with the Ring Road ground-truth vectors, `train.pkl` / `test.pkl`), `previews/` and `manifest.json`.

If the job hits its time limit or fails, run the script again; finished splits are skipped. Override resources with sbatch options, e.g. `./run_stage1_slurm.sh --gres shard:rtx_3090:16384,tmpdisk:102400` or `--partition=compute_dense --time=3-00:00:00`. `SEMVECNET_WORK` (default `~/semvecnet-phase1`, ~3 GB of assets and per-split runs) and `SEMVECNET_DATA` (default `<repo>/data`) move the outputs. `/home` has a 20 GiB quota, so bags and Docker images never go there.

## Any GPU server with Docker

From a fresh checkout on the server:

```sh
python3 tools/run_phase1.py --work-dir "$HOME/semvecnet-phase1"
```

You need Linux (x86-64), Python 3.9+, Git, Docker with the NVIDIA Container Toolkit and a compatible NVIDIA driver/GPU, and `rclone` with a Google Drive remote named `gdrive`. The remote needs access to both the shared Bags folder and the [model weights folder](https://drive.google.com/drive/folders/1bV7mnjSOYg2M-LBbM59C_eFDIvqvZZ7i). Configure it with `rclone config` before running; for a headless server, follow rclone's remote authorization instructions. Do not commit its configuration or credentials. Use your own OAuth client ID if prompted about retirement of rclone's shared Google Drive client.

Allow **80 GB** for Phase 1 on a fresh server, or **100–120 GB** if you later keep both the HRNet and SemVecNet images. `--work-dir` controls bags, assets and outputs, not Docker's storage location. If they use separate filesystems, check free space on both. Phase 1 builds the HRNet Docker image and a TensorRT engine for the server GPU. It does not pull the SemVecNet training image or start fine-tuning.

## What runs

[`tools/phase1_plan_b.txt`](../tools/phase1_plan_b.txt) is the fixed eight-split list, including Drive copies of the two May 30 files. A fresh server transfers about **154 GB** in total, but the largest individual split is **21.5 GB**. Each split is downloaded, mapped at `--map-hz 5` near the Lanelet2 map, aligned, and deleted before the next transfer. Native six-camera bags use all six cameras; eight-camera bags use all eight. The terminal shows the current split, transfer %, scan progress and export status. If a bag fails, the script stops before the next split and keeps any completed download for retry. Rerun with the same `--work-dir` to skip successful splits.

The Feb 24 daytime drive is held out for testing; the dusk, night and May 30 runs are for training. The exporter keeps roughly one frame per 30 m (15 m in dense areas), subject to alignment and coverage filters. **~250 train / ~110 test are estimates**, not quotas. The command reports the actual counts and writes per-run keep/drop counts to `outputs/dataset/manifest.json`.

## Inspect maps while the job runs

From your laptop, forward the preview port and open <http://127.0.0.1:8765/> in your browser:

```sh
ssh -N -L 8765:127.0.0.1:8765 user@server
```

- `runs/<split>/maps/` contains numbered semantic PNGs as they are written. Refresh the listing for new frames. `cams/` contains matching front camera JPEGs where available.
- `runs/<split>/offset/` contains `overlay.png` and `zoom.png` once alignment completes. `stage1.log` and `runs/batch_log.jsonl` help diagnose failures.
- `dataset/previews/train/` and `dataset/previews/test/` contain semantic maps with ground-truth vectors drawn on top. `dataset/` is the directory to mount at `/data/ringroad` when fine-tuning.

The preview server listens only on server loopback and shuts down when the script exits. To browse the outputs later over the same SSH tunnel, run `python3 -m http.server 8765 --bind 127.0.0.1 --directory "$HOME/semvecnet-phase1/outputs"` on the server. Use `--preview-port 0` to turn off previews during the run.

## Fine-tune on WATcloud

Place the exported dataset at `~/ringroad_dataset_large`, with
`train/train.pkl`, `test/test.pkl` and the corresponding maps. From the login
node, run `./fine-tune.sh`. Extra arguments are passed to sbatch. The launcher
follows `logs/finetune-<job>.out`. Ctrl-C leaves the submitted job running.

The job starts from `SemVecNet/ckpts/mAP_epoch_99.pth` and fetches it with the
configured rclone `gdrive:` remote if missing. Set `SEMVECNET_DATASET` or
`SEMVECNET_CKPT` to use different inputs. Defaults are 40 epochs, batch size 4,
4 loader workers and AdamW at 0.0001. For example, `BATCH=2 EPOCHS=10 ./fine-tune.sh`
reduces the batch size and duration. Set `AUGMENT=0` to disable training flips,
rotations and translations. Evaluation always uses the original maps.

Epoch 0 measures the starting weights. During training, `[TRAIN]` reports loss,
learning rate and time left in the epoch. Each `[SUMMARY]` gives mean training
loss, Chamfer mAP on the same 150 training maps (or all if smaller), full test
mAP and the four test class APs averaged over 0.5, 1.0 and 1.5 metres.

Results go to `~/semvecnet-finetune/<job-id>/`. Set `SEMVECNET_FT_OUT` to
select a fresh output directory. It contains the two ground-truth JSON caches,
`metrics.csv`, `metrics.json`, and weights-only `latest.pth` and `best.pth`.
At completion `ringroad_finetuned.pth` is a hard link to the highest-test-mAP
checkpoint, including epoch 0 if fine-tuning never improves it. Pass this file
as the checkpoint for `tools/mcap_infer_semvecnet.py`.
