#!/usr/bin/env python3
"""Progress of a tools/batch_stage1.py run: one overall bar (weighted by seconds of recording, which is what stage 1
time scales with) and one line per run.

  python3 tools/batch_progress.py [RUNS_DIR] [--watch [SECONDS]]

RUNS_DIR (default ~/Desktop/data/semvecnet_runs/finetune) holds bags.txt, batch.out and the run directories.
Bag durations come from the MCAP summary of the local / downloaded file, else from RUNS_DIR/bag_info.json.
The ETA covers stage 1 + offsets at the speed of the runs finished so far; downloads (when not prefetched) add to it.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_stage1 import mcap_seconds, parse_list, run_name  # noqa: E402

FRAME = re.compile(r'(?:frame\s+\d+:|scan)\s+t\+\s*([\d.]+)s')


def bar(frac, width=30):
    frac = min(max(frac, 0.0), 1.0)
    full = int(frac * width)
    half = frac * width - full >= 0.5
    return '█' * full + ('▌' if half else '') + '░' * (width - full - half)


def fmt_t(s):
    s = int(s)
    return f'{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}'


def snapshot(runs, cache):
    jobs = parse_list(runs / 'bags.txt')
    info = json.loads((runs / 'bag_info.json').read_text()) if (runs / 'bag_info.json').exists() else {}
    log = (runs / 'batch.out').read_text(errors='replace') if (runs / 'batch.out').exists() else ''
    # stage-1 output of each run in batch.out: from its "$ ... -o <run dir>" line to the next run's
    sections, cur = {}, None
    for line in log.splitlines():
        m = re.search(r' -o (\S+)', line) if line.startswith('  $ ') else None
        if m:
            cur = Path(m.group(1)).name
            sections[cur] = []
        elif cur:
            sections[cur].append(line)

    rows, total = [], 0.0
    done_s = done_wall = 0.0
    for split, sets in jobs.items():
        local = Path(split).expanduser()
        f = local if local.is_file() else cache / Path(split).name
        secs = mcap_seconds(f) if f.is_file() else None
        secs = secs or info.get(split, {}).get('seconds') or info.get(str(local), {}).get('seconds') or 0.0
        nbytes = info.get(split, {}).get('bytes')
        for camset, _ in sets:
            name = run_name(split, camset)
            out = runs / name
            total += secs
            if (out / 'DONE').exists():
                rec = json.loads((out / 'DONE').read_text())
                done_s += secs
                done_wall += rec.get('stage1_s', 0)
                u = rec.get('usable') or {}
                extra = f'{rec.get("keyframes", 0)} keyframes'
                if rec.get('offset'):
                    extra += f', offset {rec["offset"][0]:+.2f} {rec["offset"][1]:+.2f} m [{rec.get("verdict")}]'
                if u:
                    extra += f', {u.get("keyframes")} usable ({100 * u.get("share", 0):.0f}%)'
                rows.append(('done', name, 1.0, extra, secs))
                continue
            lines = sections.get(name)
            failed = re.search(rf'-> {re.escape(name)}: (stage 1 failed|download failed|offset failed)', log)
            if failed:
                rows.append(('FAILED', name, 0.0, failed.group(1) + ' - see batch.out', secs))
            elif lines is not None:
                ts = [float(m.group(1)) for l in lines if (m := FRAME.search(l))]
                t = ts[-1] if ts else 0.0
                frac = min(t / secs, 0.99) if secs else 0.0
                phase = 'offsets' if any(l.startswith('Done:') for l in lines) else 'stage 1'
                rows.append(('running', name, frac, f'{phase}, t+{t:.0f}/{secs:.0f} s of recording', secs))
            else:
                part = None
                if not local.is_file() and cache.is_dir():
                    # rclone writes <name>.<hash>.partial while downloading (preallocated: count allocated blocks)
                    for p in [cache / Path(split).name] + sorted(cache.glob(Path(split).name + '*.partial')):
                        if p.exists():
                            part = p.stat().st_blocks * 512 if p.suffix == '.partial' else p.stat().st_size
                dl = ''
                if part:
                    dl = 'downloading ' if part < (nbytes or float('inf')) else 'downloaded '
                    dl += f'{part / 1e9:.1f}' + (f'/{nbytes / 1e9:.1f} GB' if nbytes else ' GB')
                rows.append(('queued', name, 0.0, dl, secs))
    progress = sum(r[2] * r[4] for r in rows)
    return rows, total, progress, done_s, done_wall


def render(runs, cache):
    rows, total, progress, done_s, done_wall = snapshot(runs, cache)
    frac = progress / total if total else 0.0
    n_done = sum(r[0] == 'done' for r in rows)
    eta = ''
    if done_s > 0 and done_wall > 0:
        eta = f'   ETA ~{fmt_t((total - progress) * done_wall / done_s)} (stage 1 + offsets; downloads extra)'
    out = [f'Stage 1 batch  [{bar(frac)}] {100 * frac:3.0f}%   {n_done}/{len(rows)} runs done{eta}', '']
    for status, name, f, extra, secs in rows:
        b = f'[{bar(f, 12)}] {100 * f:3.0f}%' if status == 'running' else ' ' * 19
        out.append(f'  {status:8s} {name:48s} {b}  {extra}')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='?', default=str(Path.home() / 'Desktop/data/semvecnet_runs/finetune'))
    ap.add_argument('--cache', default=str(Path.home() / 'Desktop/data/bag_cache'))
    ap.add_argument('--watch', nargs='?', const=5.0, type=float, help='refresh every N seconds (default 5)')
    args = ap.parse_args()
    runs, cache = Path(args.runs).expanduser(), Path(args.cache).expanduser()
    if not args.watch:
        print(render(runs, cache))
        return
    try:
        while True:
            text = render(runs, cache)
            os.system('clear')
            print(text + f'\n\n  {time.strftime("%H:%M:%S")}  (Ctrl-C stops watching; the batch keeps running)')
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
