#!/usr/bin/env python3
"""Download, map, align, and export the Ring Road Plan B dataset; do not train."""
import argparse
from contextlib import contextmanager
import fcntl
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Thread

from batch_stage1 import REPO, parse_list, run_name

TOOLS = REPO / 'tools'
PLAN = TOOLS / 'phase1_plan_b.txt'
OSM = REPO / 'ring_road.osm'
IMAGE = 'semvecnet-mcap-hrnet:phase1'
TEST_DRIVE = 'recording_20260224_213947_*'


class PreviewHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


@contextmanager
def previews(root, port):
    if port == 0:
        yield
        return
    with ThreadingHTTPServer(('127.0.0.1', port), partial(PreviewHandler, directory=root)) as http:
        thread = Thread(target=http.serve_forever, daemon=True)
        thread.start()
        try:
            print(f'Live previews: http://127.0.0.1:{port}/ (use SSH port forwarding from your laptop)', flush=True)
            yield
        finally:
            http.shutdown()
            thread.join()


def run_phase1(work):
    root = work / 'outputs'
    runs, dataset = root / 'runs', root / 'dataset'
    runs.mkdir(parents=True, exist_ok=True)
    dataset.mkdir(exist_ok=True)
    expected = {run_name(split, camset) for split, jobs in parse_list(PLAN).items() for camset, _ in jobs}
    extra = {p.parent.name for p in runs.glob('*/DONE')} - expected
    if extra:
        raise RuntimeError(f'Unrelated completed runs in {runs}: {", ".join(sorted(extra))}. Use a dedicated --work-dir.')
    env = dict(os.environ, SEMVECNET_HRNET_IMAGE=IMAGE, SEMVECNET_ASSETS=str(work / 'assets'))

    for cmd in ('docker', 'rclone', 'git'):
        if not shutil.which(cmd):
            raise RuntimeError(f'{cmd} is required but was not found on PATH')
    if 'gdrive:' not in subprocess.check_output(['rclone', 'listremotes'], text=True).splitlines():
        raise RuntimeError('Configure rclone remote gdrive with access to the shared Bags and weights folders.')
    subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'], check=True)
    print('[Setup] Preparing the HRNet image and GPU-specific TensorRT engine', flush=True)
    subprocess.run([REPO / 'run_mcap_pipeline.sh', '--prepare-stage1'], env=env, check=True)

    print('[Bags] Processing Plan B in order (one MCAP split on disk)', flush=True)
    subprocess.run([sys.executable, TOOLS / 'batch_stage1.py', PLAN, '--runs', runs, '--cache', work / 'cache',
                    '--osm', OSM, '--offset-image', IMAGE, '--live'], env=env, check=True)

    print('[Export] Generating ground-truth labels, train/test sets, and overlay previews', flush=True)
    subprocess.run(['docker', 'run', '--rm', '--entrypoint', 'python', '--user', f'{os.getuid()}:{os.getgid()}',
                    '-e', 'HOME=/tmp', '-e', 'PYTHONDONTWRITEBYTECODE=1',
                    '-v', f'{TOOLS}:/repo/tools:ro', '-v', f'{OSM}:/gt/map.osm:ro',
                    '-v', f'{runs}:/runs:ro', '-v', f'{dataset}:/dataset', IMAGE,
                    '/repo/tools/build_finetune_dataset.py', '--runs', '/runs',
                    '--osm', '/gt/map.osm', '--out', '/dataset', '--test', TEST_DRIVE], check=True)
    totals = json.loads((dataset / 'manifest.json').read_text())['totals']
    for split in ('train', 'test'):
        if not totals[split]['maps'] or not (dataset / split / f'{split}.pkl').is_file():
            raise RuntimeError(f'{split} export is empty; check per-run filtering in {dataset / "manifest.json"}.')
    print(f'[Done] {totals["train"]["maps"]} train / {totals["test"]["maps"]} test maps '
          f'(planning estimates: ~250 / ~110). Dataset: {dataset}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', type=Path, default=Path.home() / 'semvecnet-phase1',
                    help='dedicated server disk directory for assets, the transient bag, and outputs')
    ap.add_argument('--preview-port', type=int, default=8765, help='loopback HTTP port (0 disables live previews)')
    args = ap.parse_args()
    if not 0 <= args.preview_port <= 65535:
        ap.error('--preview-port must be between 0 and 65535')
    work = args.work_dir.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    root = work / 'outputs'
    root.mkdir(exist_ok=True)
    try:
        with (work / '.phase1.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                sys.exit(f'Phase 1 is already running in {work}')
            with previews(root, args.preview_port):
                run_phase1(work)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(f'Phase 1 stopped: {exc}')
    except KeyboardInterrupt:
        sys.exit('Phase 1 interrupted; rerun with the same --work-dir to resume.')


if __name__ == '__main__':
    main()
