#!/usr/bin/env python3
"""Batch driver for fine-tuning data: download each bag split, run stage 1 (semantic maps) on it once per camera set,
estimate its pose/map offset, delete the split. Runs on the host (needs rclone and docker); stdlib only.

  python3 tools/batch_stage1.py bags.txt [--runs DIR] [--cache DIR] [--keep] [--no-offset]

bags.txt - one line per (split, camera set); blank lines and # comments ignored:
  <split path>  <camera set>  [extra stage-1 options...]
    split path   path inside the Drive "Bags" folder (downloaded with rclone, remote "gdrive"), or a local .mcap path
    camera set   pano8 | pano6 | all | a comma list of camera names (camera_pano_nn,...)
                 pano8 = the eight roof cameras, pano6 = pano8 without camera_pano_ee / camera_pano_ww
  Example:
    recording_20260224_213947/recording_20260224_213947_0.mcap  pano8
    recording_20260224_213947/recording_20260224_213947_0.mcap  pano6
    recording_20261004_185825/recording_20261004_185825_1.mcap  pano8  --max-cam-dt 0.15

Each split is downloaded once and every camera set listed for it is run before it is deleted, so only one split is
on disk at a time (--prefetch downloads the next one during processing: faster, two splits on disk).
Output: <runs>/<split stem>__<camera set>/ (stage-1 run + offset/), a DONE file when finished (re-running the
batch skips those), and <runs>/batch_log.jsonl with one line per run.
Every split is processed on its own, so the first seconds of each split have a sparse map (nothing accumulated yet);
tools/build_finetune_dataset.py drops them (--warmup).
"""
import argparse
import json
import os
import queue
import shlex
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BAGS_FOLDER_ID = '127Kw3o7Org474rkK1wwMDRZKiYHaR0Db'  # Drive folder "Bags"
SVN_IMAGE = 'naruarjun/semvecnet:v2'
PANO8 = ['camera_pano_nn', 'camera_pano_ne', 'camera_pano_ee', 'camera_pano_se',
         'camera_pano_ss', 'camera_pano_sw', 'camera_pano_ww', 'camera_pano_nw']
PRESETS = dict(pano8=PANO8, pano6=[c for c in PANO8 if c not in ('camera_pano_ee', 'camera_pano_ww')], all=None)


def parse_list(path):
    jobs = OrderedDict()  # split -> [(camera set, extra args)]
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        parts = shlex.split(line)
        if len(parts) < 2:
            sys.exit(f'{path}:{n}: need "<split path> <camera set> [options]"')
        jobs.setdefault(parts[0], []).append((parts[1], parts[2:]))
    return jobs


def run_name(split, camset):
    tag = camset if camset in PRESETS else 'cams' + str(len(camset.split(',')))
    return f'{Path(split).stem}__{tag}'


def mcap_seconds(path):
    """Read the recording duration from the MCAP summary, without scanning the bag."""
    try:
        with path.open('rb') as f:
            footer = f.seek(-37, os.SEEK_END)
            tail = f.read(37)
            if tail[-8:] != b'\x89MCAP0\r\n' or tail[0] != 0x02:
                return None
            start = struct.unpack_from('<Q', tail, 9)[0]
            if not 0 < start < footer:
                return None
            f.seek(start)
            while f.tell() + 9 <= footer:
                op, size = struct.unpack('<BQ', f.read(9))
                if f.tell() + size > footer:
                    return None
                if op == 0x0B and size >= 42:
                    t0, t1 = struct.unpack_from('<QQ', f.read(42), 26)
                    return (t1 - t0) / 1e9
                f.seek(size, os.SEEK_CUR)
    except (OSError, struct.error):
        pass
    return None


def progress(label, fraction, detail=''):
    n = int(max(0, min(fraction, 1)) * 24)
    print(f'[{("=" * n).ljust(24, ".")}] {label} {detail}'.rstrip(), flush=True)


def download(split, cache, live=False):
    local = Path(split).expanduser()
    if local.is_file():
        return local, False
    dst = cache / Path(split).name
    if dst.is_file():
        return dst, True
    cache.mkdir(parents=True, exist_ok=True)
    t = time.time()
    cmd = ['rclone', 'copyto', '--drive-root-folder-id', BAGS_FOLDER_ID, f'gdrive:{split}', str(dst),
           '--multi-thread-streams', '8']
    if live:
        # Native transfer progress includes bytes, speed, ETA and a progress bar.
        cmd += ['--progress', '--stats', '5s']
    r = subprocess.run(cmd, stderr=None if live else subprocess.PIPE, text=True)
    if r.returncode != 0 or not dst.is_file():
        err = '\n'.join(l for l in (r.stderr or '').splitlines() if 'NOTICE' not in l)
        raise RuntimeError(f'rclone did not produce {dst.name} (exit {r.returncode}): {err[-500:]}')
    print(f'  downloaded {dst.name} ({dst.stat().st_size / 1e9:.1f} GB, {time.time() - t:.0f}s)', flush=True)
    return dst, True


def stage1(bag, out, camset, extra, osm, live=False, keep_inputs=False):
    cams = PRESETS[camset] if camset in PRESETS else camset.split(',')
    cmd = [str(REPO / 'run_mcap_pipeline.sh'), str(bag), '-o', str(out), '--stages', '1', '--gt', str(osm)] + \
          (['--cameras', ','.join(cams)] if cams else []) + extra
    if not keep_inputs:
        cmd += ['--no-inputs']
    print('  $ ' + ' '.join(shlex.quote(c) for c in cmd), flush=True)
    if not live:
        return subprocess.run(cmd).returncode
    out.mkdir(parents=True, exist_ok=True)
    duration = mcap_seconds(bag)
    with (out / 'stage1.log').open('w') as log, subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ) as proc:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if line.startswith('  scan t+') and duration:
                elapsed = float(line.removeprefix('  scan t+').strip().removesuffix('s'))
                progress('Scanning MCAP', min(0.99, elapsed / duration), f'{elapsed:.0f}/{duration:.0f} s')
            else:
                print(line, end='', flush=True)
        return proc.wait()


def offsets(out, osm, image=SVN_IMAGE):
    cmd = ['docker', 'run', '--rm', '--entrypoint', 'python', '--user', f'{os.getuid()}:{os.getgid()}', '-e', 'HOME=/tmp',
           '-e', 'PYTHONWARNINGS=ignore', '-v', f'{REPO / "tools"}:/repo/tools:ro',
           '-v', f'{Path(osm).resolve()}:/gt/map.osm:ro', '-v', f'{out}:/run', image,
           '/repo/tools/mcap_estimate_offset.py', '--run', '/run', '--osm', '/gt/map.osm']
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for l in r.stdout.splitlines():
        if r.returncode or l.startswith('  '):
            print(l, flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('list', help='bag list (see above)')
    ap.add_argument('--runs', default=str(Path.home() / 'Desktop/data/semvecnet_runs/finetune'), help='output root')
    ap.add_argument('--cache', default=str(Path.home() / 'Desktop/data/bag_cache'), help='where splits are downloaded')
    ap.add_argument('--osm', default=str(REPO / 'ring_road.osm'))
    ap.add_argument('--keep', action='store_true', help='keep downloaded splits')
    ap.add_argument('--no-offset', action='store_true', help='skip the offset estimation')
    ap.add_argument('--prefetch', action='store_true', help='download the next split while the current one is '
                    'processed (two splits on disk at once; default: one at a time)')
    ap.add_argument('--keep-inputs', action='store_true', help="keep stage 1's inputs/ (every camera image + LiDAR "
                     'sweep per keyframe, ~2 GB per split, only used by tools/mcap_compare_viz.py)')
    ap.add_argument('--live', action='store_true', help='display download and scan progress; write per-run stage1.log')
    ap.add_argument('--offset-image', default=SVN_IMAGE)
    args = ap.parse_args()
    if args.live and args.prefetch:
        ap.error('--live requires sequential downloads (omit --prefetch)')

    jobs = parse_list(args.list)
    runs, cache = Path(args.runs).expanduser(), Path(args.cache).expanduser()
    runs.mkdir(parents=True, exist_ok=True)
    todo = [(s, v) for s, v in jobs.items() if not all((runs / run_name(s, c) / 'DONE').exists() for c, _ in v)]
    print(f'{len(jobs)} splits, {sum(len(v) for v in jobs.values())} runs; {len(todo)} splits still to do', flush=True)
    if not args.keep:
        # Recover if interrupted after DONE but before removing the cached download.
        pending = {s for s, _ in todo}
        for s in jobs.keys() - pending:
            if not Path(s).expanduser().is_file():
                (cache / Path(s).name).unlink(missing_ok=True)

    def fetch(s):
        try:
            return (s, *download(s, cache, args.live), None)
        except Exception as e:
            return (s, None, False, e)

    if args.prefetch:  # split i+1 downloads while split i is processed
        q = queue.Queue(maxsize=1)

        def downloader():
            for s, _ in todo:
                q.put(fetch(s))
            q.put(None)
        threading.Thread(target=downloader, daemon=True).start()
        items = iter(q.get, None)
    else:
        def serial():
            for i, (s, _) in enumerate(todo, len(jobs) - len(todo) + 1):
                if args.live:
                    progress(f'[{i}/{len(jobs)}] Downloading / checking cache:', (i - 1) / len(jobs), Path(s).name)
                yield fetch(s)
        items = serial()
    log = runs / 'batch_log.jsonl'
    for split, bag, downloaded, err in items:
        print(f'== {split}', flush=True)
        for camset, extra in jobs[split]:
            name = run_name(split, camset)
            out = runs / name
            if (out / 'DONE').exists():
                continue
            rec = dict(split=split, camera_set=camset, run=name, extra=extra, time=time.strftime('%Y-%m-%d %H:%M:%S'))
            if err is not None:
                rec.update(status='download failed', error=str(err))
            else:
                t = time.time()
                print(f'  Preprocessing {name}; map previews: {out / "maps"}', flush=True)
                rc = stage1(bag, out, camset, extra, args.osm, args.live, args.keep_inputs)
                n = sum(1 for _ in open(out / 'frames.jsonl')) if (out / 'frames.jsonl').exists() else 0
                rec.update(stage1_rc=rc, keyframes=n, stage1_s=round(time.time() - t))
                if rc != 0 or n == 0:
                    rec['status'] = 'stage 1 failed'
                else:
                    if not args.no_offset:
                        print(f'  Estimating pose offsets for {name}', flush=True)
                        rec['offset_rc'] = offsets(out, args.osm, args.offset_image)
                        oj = out / 'offset' / 'offset.json'
                        if rec['offset_rc'] == 0 and oj.is_file():
                            o = json.loads(oj.read_text())
                            rec.update(offset=o['run_offset'], verdict=o['verdict'], usable=o['usable'])
                        else:
                            rec['status'] = 'offset failed'
                    if 'status' not in rec:
                        if not args.keep_inputs and (out / 'inputs').is_dir():
                            shutil.rmtree(out / 'inputs')
                        rec['status'] = 'done'
                        tmp = out / 'DONE.tmp'
                        tmp.write_text(json.dumps(rec, indent=1))
                        tmp.replace(out / 'DONE')
            print(f'  -> {name}: {rec["status"]}', flush=True)
            with open(log, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            if rec['status'] != 'done':
                sys.exit(f'{name}: {rec["status"]}; stopped before the next split. See {log}.')
        if downloaded and bag is not None and not args.keep:
            bag.unlink(missing_ok=True)
            print(f'  Deleted cached split {bag.name}', flush=True)
    if args.live:
        progress(f'[{len(jobs)}/{len(jobs)}] All bag splits complete', 1)
    print(f'Finished. Log: {log}', flush=True)


if __name__ == '__main__':
    main()
