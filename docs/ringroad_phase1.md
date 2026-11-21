# Ring Road Phase 1 on a GPU server

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
