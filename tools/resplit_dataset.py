#!/usr/bin/env python3
"""Re-split a fine-tuning dataset by run (train / test) and balance the training conditions.

  python tools/resplit_dataset.py --src ringroad_dataset_clean --out ringroad_dataset_mixed \
      --test 'recording_20260224_213947_1__*' 'recording_20260217_224728_1_with_eidos__*' \
      --repeat 'may_30_*=2' 'pano_lidar_cc_20261029_*=2' --snow 'recording_20260217_*' 'recording_20260224_*'

  --test     run-name patterns that form the test set; every other run is training data. Runs are whole bag splits;
             splits of one drive cover different stretches of road, so a held-out split is unseen road.
  --repeat   PATTERN=N: training runs matching PATTERN appear N times per epoch (each copy gets its own token and
             a different random augmentation), so rare conditions (summer day, night) weigh as much as common ones
  --snow     run-name patterns recorded with snow on the ground (for the condition table in manifest.json)

Maps and labels are copied unchanged; <split>.pkl and manifest.json are rewritten.
"""
import argparse
import fnmatch
import json
import pickle
import shutil
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--test', nargs='+', required=True)
    ap.add_argument('--repeat', nargs='*', default=[])
    ap.add_argument('--snow', nargs='*', default=[])
    args = ap.parse_args()
    if args.out.resolve() == args.src.resolve():
        ap.error('--out must differ from --src')
    repeat = [(p.rsplit('=', 1)[0], int(p.rsplit('=', 1)[1])) for p in args.repeat]

    match = lambda run, pats: any(fnmatch.fnmatch(run, p) for p in pats)
    infos, metadata = [], None
    for split in ('train', 'test'):
        d = pickle.load(open(args.src / split / f'{split}.pkl', 'rb'))
        metadata = d['metadata']
        infos += [(split, i) for i in d['infos']]
    runs = sorted({i['scene_token'] for _, i in infos})
    unknown = [p for p in args.test + [p for p, _ in repeat] + args.snow if not any(fnmatch.fnmatch(r, p) for r in runs)]
    if unknown:
        raise SystemExit(f'patterns matching no run: {unknown}\nruns: {runs}')

    if args.out.exists():
        shutil.rmtree(args.out)
    out = {'train': [], 'test': []}
    for src_split, info in infos:
        run = info['scene_token']
        split = 'test' if match(run, args.test) else 'train'
        for rel in (info['sem_path'], 'labels/' + info['sem_path'][5:-4] + '.json'):
            (args.out / split / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.src / src_split / rel, args.out / split / rel)
        n = next((k for p, k in repeat if fnmatch.fnmatch(run, p)), 1) if split == 'train' else 1
        for c in range(n):
            out[split].append(dict(info, token=info['token'] if c == 0 else f'{info["token"]}_rep{c}'))

    manifest = json.loads((args.src / 'manifest.json').read_text())
    manifest['resplit'] = dict(source=str(args.src), test=args.test, repeat=args.repeat, snow=args.snow)
    cond = lambda i: f'{i["light"]}, {"snow" if match(i["scene_token"], args.snow) else "no snow"}'
    totals = {}
    for split, lst in out.items():
        lst = [dict(i, frame_idx=k) for k, i in enumerate(lst)]
        with open(args.out / split / f'{split}.pkl', 'wb') as fh:
            pickle.dump(dict(infos=lst, metadata=metadata), fh, protocol=4)
        unique = [i for i in lst if '_rep' not in i['token']]
        totals[split] = dict(maps=len(unique), samples_per_epoch=len(lst),
                             runs=sorted({i['scene_token'] for i in lst}),
                             conditions_maps=dict(Counter(cond(i) for i in unique)),
                             conditions_per_epoch=dict(Counter(cond(i) for i in lst)))
        print(f'{split}: {len(unique)} maps ({len(lst)} samples per epoch)')
        for k in sorted(totals[split]['conditions_maps']):
            print(f'   {k:18s} {totals[split]["conditions_maps"][k]:4d} maps   {totals[split]["conditions_per_epoch"][k]:4d} per epoch')
        for r in totals[split]['runs']:
            n = sum(i['scene_token'] == r for i in unique)
            print(f'     {r:48s} {n:4d}  x{sum(i["scene_token"] == r for i in lst) // max(n, 1)}')
    manifest['totals_before_resplit'] = manifest.get('totals')
    manifest['totals'] = totals
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
