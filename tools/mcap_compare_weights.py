#!/usr/bin/env python3
"""Old vs new SemVecNet weights on the same stage-1 run: one side-by-side video and a metrics table.

Both runs must have gone through run_mcap_pipeline.sh --stages 23 (same semantic maps, same ground truth and pose
offset). Each frame reuses the panels mcap_compare_viz.py drew for the two runs:

    +----------------------------- cameras (raw images of the keyframe) -----------------------------+
    | semantic map | OLD SemVecNet | NEW SemVecNet | ground truth | OLD overlay | NEW overlay          |

    python tools/mcap_compare_weights.py --old <run> --new <run> --out <dir>
      -> <dir>/old_vs_new/<frame>.png, <dir>/old_vs_new.mp4, <dir>/metrics_old_vs_new.{json,md}
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MAP_CLASSES = ['divider', 'ped_crossing', 'boundary', 'centerline']
PW, PH = 360, 720          # panel size of tools/mcap_compare_viz.py
BG = (24, 24, 24)
OLD_BGR, NEW_BGR = (120, 170, 255), (120, 255, 140)


def relabel(panel, title, sub, colour):
    p = panel.copy()
    cv2.rectangle(p, (0, 0), (p.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(p, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1, cv2.LINE_AA)
    cv2.putText(p, sub, (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--old', required=True, type=Path)
    ap.add_argument('--new', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()

    # ---------------------------------------------------------------- metrics table
    mo, mn = (json.loads((r / 'eval' / 'metrics.json').read_text()) for r in (args.old, args.new))
    om, nm = mo['raw']['mapped'], mn['raw']['mapped']
    rows = [('mAP (mapped roads)', om['mAP'], nm['mAP']),
            ('mAP (whole 60x30 m window)', mo['raw']['full']['mAP'], mn['raw']['full']['mAP'])]
    rows += [(f'{c} AP', om['AP'][c], nm['AP'][c]) for c in MAP_CLASSES]
    for c in MAP_CLASSES:
        o, n = om['at_score_thr']['per_class'][c], nm['at_score_thr']['per_class'][c]
        rows += [(f'{c} precision', o['precision'], n['precision']), (f'{c} recall', o['recall'], n['recall'])]
    note = (f'{mo["frames_scored"]} keyframes on the map. AP = MapTR chamfer AP averaged over 0.5/1.0/1.5 m against '
            f'the Lanelet2 map (all scores); precision/recall at 1.0 m and the drawing thresholds '
            f'{om["at_score_thr"]["score_thr"]}. Ground-truth instances: {om["num_gt"]}')
    md = ['| metric | old | new | change |', '|---|---|---|---|']
    md += [f'| {k} | {o:.3f} | {n:.3f} | {n - o:+.3f} |' for k, o, n in rows]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'metrics_old_vs_new.md').write_text('\n'.join(md + ['', note]) + '\n')
    (args.out / 'metrics_old_vs_new.json').write_text(json.dumps(
        dict(metrics={k: dict(old=o, new=n) for k, o, n in rows}, frames_scored=mo['frames_scored'],
             old_checkpoint=str(args.old), new_checkpoint=str(args.new)), indent=2))
    w = max(len(k) for k, _, _ in rows)
    print(f'[RESULT] {"":{w}}       old       new   change')
    for k, o, n in rows:
        print(f'[RESULT] {k:{w}}  {o:8.3f}  {n:8.3f}  {n - o:+7.3f}')
    print(f'[RESULT] {note}')

    # ---------------------------------------------------------------- side-by-side frames
    frames = sorted(p.name for p in (args.old / 'compare').glob('*.png'))
    (args.out / 'old_vs_new').mkdir(exist_ok=True)
    first = next((json.loads(l) for l in (args.old / 'frames.jsonl').read_text().splitlines() if l.strip()), {})
    fps = float(first.get('fps', 2.0))
    width = 6 * PW
    writer, size, n = None, None, 0
    for name in frames:
        a, b = cv2.imread(str(args.old / 'compare' / name)), cv2.imread(str(args.new / 'compare' / name))
        if a is None or b is None:
            continue
        bev_a, bev_b = a[-PH:], b[-PH:]   # panels: 0 LiDAR, 1 semantic map, 2 SemVecNet, 3 ground truth, 4 overlay

        def panel(bev, k):
            return bev[:, k * PW:(k + 1) * PW]
        row = np.hstack([
            panel(bev_a, 1),
            relabel(panel(bev_a, 2), 'OLD  SemVecNet', f'mAP {om["mAP"]:.3f} on this bag', OLD_BGR),
            relabel(panel(bev_b, 2), 'NEW  SemVecNet', f'mAP {nm["mAP"]:.3f} on this bag', NEW_BGR),
            panel(bev_a, 3),
            relabel(panel(bev_a, 4), 'OVERLAY  OLD', 'GT wide/faded, OLD thin', OLD_BGR),
            relabel(panel(bev_b, 4), 'OVERLAY  NEW', 'GT wide/faded, NEW thin', NEW_BGR),
        ])
        for k in range(1, 6):
            cv2.line(row, (k * PW, 0), (k * PW, PH), (90, 90, 90), 1)
        top = a[:-PH]                       # frame header, cameras, legend
        top = cv2.copyMakeBorder(top, 0, 0, 0, width - top.shape[1], cv2.BORDER_CONSTANT, value=BG)
        img = np.vstack([top, row])
        cv2.imwrite(str(args.out / 'old_vs_new' / name), img)
        if writer is None:
            size = (img.shape[1], img.shape[0])
            writer = cv2.VideoWriter(str(args.out / 'old_vs_new.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        writer.write(img if (img.shape[1], img.shape[0]) == size else cv2.resize(img, size))
        n += 1
    if writer is not None:
        writer.release()
    print(f'Wrote {n} side-by-side frames to {args.out / "old_vs_new"} and {args.out / "old_vs_new.mp4"}')


if __name__ == '__main__':
    main()
