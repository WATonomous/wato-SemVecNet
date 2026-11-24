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

import re

from batch_stage1 import REPO, docker_user, offsets, parse_list, run_name

TOOLS = REPO / 'tools'
PLAN = Path(os.environ.get('SEMVECNET_PLAN', TOOLS / 'phase1_plan_all.txt'))
OSM = REPO / 'ring_road.osm'
IMAGE = 'semvecnet-mcap-hrnet:phase1'


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


def run_phase1(work, cache=None, dataset=None):
    root = work / 'outputs'
    runs = root / 'runs'
    dataset = dataset or root / 'dataset'
    cache = cache or work / 'cache'
    runs.mkdir(parents=True, exist_ok=True)
    dataset.mkdir(parents=True, exist_ok=True)
    expected = {run_name(split, camset) for split, jobs in parse_list(PLAN).items() for camset, _ in jobs}
    extra = {p.parent.name for p in runs.glob('*/DONE')} - expected
    if extra:  # runs from an older plan are kept and exported too
        print(f'[Runs] {len(extra)} finished runs not in {PLAN.name} are included: {", ".join(sorted(extra))}', flush=True)
    env = dict(os.environ, SEMVECNET_HRNET_IMAGE=IMAGE, SEMVECNET_ASSETS=str(work / 'assets'))

    for cmd in ('docker', 'rclone', 'git'):
        if not shutil.which(cmd):
            raise RuntimeError(f'{cmd} is required but was not found on PATH')
    if 'gdrive:' not in subprocess.check_output(['rclone', 'listremotes'], text=True).splitlines():
        raise RuntimeError('Configure rclone remote gdrive with access to the shared Bags and weights folders.')
    subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'], check=True)
    print('[Setup] Preparing the HRNet image and GPU-specific TensorRT engine', flush=True)
    subprocess.run([REPO / 'run_mcap_pipeline.sh', '--prepare-stage1'], env=env, check=True)

    print(f'[Bags] Processing {PLAN.name} in order (one MCAP split on disk)', flush=True)
    subprocess.run([sys.executable, TOOLS / 'batch_stage1.py', PLAN, '--runs', runs, '--cache', cache,
                    '--osm', OSM, '--offset-image', IMAGE, '--live', '--keep-going'], env=env, check=True)

    # offsets from an older estimator (its fit decides which keyframes are kept) are recomputed: no bag needed
    version = int(re.search(r'^VERSION = (\d+)', (TOOLS / 'mcap_estimate_offset.py').read_text(), re.M).group(1))
    for done in sorted(runs.glob('*/DONE')):
        run, oj = done.parent, done.parent / 'offset' / 'offset.json'
        try:
            stale = json.loads(oj.read_text()).get('version') != version
        except (OSError, ValueError):
            stale = True
        if stale:
            print(f'[Offsets] Re-estimating {run.name} (estimator v{version})', flush=True)
            offsets(run, OSM, IMAGE)

    print('[Export] Generating ground-truth labels, train/test sets, and overlay previews', flush=True)
    subprocess.run(['docker', 'run', '--rm', '--entrypoint', 'python', *docker_user(),
                    '-e', 'HOME=/tmp', '-e', 'PYTHONDONTWRITEBYTECODE=1',
                    '-v', f'{TOOLS}:/repo/tools:ro', '-v', f'{OSM}:/gt/map.osm:ro',
                    '-v', f'{runs}:/runs:ro', '-v', f'{dataset}:/dataset', IMAGE,
                    '/repo/tools/build_finetune_dataset.py', '--runs', '/runs',
                    '--osm', '/gt/map.osm', '--out', '/dataset'], check=True)
    totals = json.loads((dataset / 'manifest.json').read_text())['totals']
    for split in ('train', 'test'):
        if not totals[split]['maps'] or not (dataset / split / f'{split}.pkl').is_file():
            raise RuntimeError(f'{split} export is empty; check per-run filtering in {dataset / "manifest.json"}.')
    print(f'[Done] {totals["train"]["maps"]} train / {totals["test"]["maps"]} test maps '
          f'Dataset: {dataset}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', type=Path, default=Path.home() / 'semvecnet-phase1',
                    help='dedicated server disk directory for assets, the transient bag, and outputs')
    ap.add_argument('--cache', type=Path, help='where each bag split is downloaded (default: <work-dir>/cache); '
                    'needs room for the largest split, ~22 GB')
    ap.add_argument('--dataset-out', type=Path, help='where the train/test dataset is written '
                    '(default: <work-dir>/outputs/dataset)')
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
                run_phase1(work, args.cache and args.cache.expanduser().resolve(),
                           args.dataset_out and args.dataset_out.expanduser().resolve())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(f'Phase 1 stopped: {exc}')
    except KeyboardInterrupt:
        sys.exit('Phase 1 interrupted; rerun with the same --work-dir to resume.')


if __name__ == '__main__':
    main()
