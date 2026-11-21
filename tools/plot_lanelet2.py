#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = ["numpy", "shapely", "pyproj", "networkx", "opencv-python-headless", "matplotlib", "scipy"]
# ///
"""Plot a Lanelet2 map (e.g. ring_road.osm): what is in the file, and the SemVecNet ground truth derived from it.

    uv run tools/plot_lanelet2.py ring_road.osm                      # -> ring_road_lanelet2.png
    uv run tools/plot_lanelet2.py ring_road.osm --run ~/Desktop/data/semvecnet_runs/<bag>   # + ego path of a run
    uv run tools/plot_lanelet2.py ring_road.osm --show               # interactive window (zoom / pan)

Left:  raw Lanelet2 - road lanelets (grey), crosswalk lanelets (blue), lanelet bounds, stop lines (red),
       traffic lights (yellow) and stop signs (magenta); --labels adds lanelet names / ids.
Right: what tools/mcap_eval_lanelet2.py scores against - divider, ped_crossing, boundary, centerline,
       plus the junction areas where dividers are dropped.
Coordinates are metres in the map's UTM zone, relative to the map's lower-left corner.
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcap_eval_lanelet2 import MAP_CLASSES, WorldGT, find_junctions, load_lanelet2  # noqa: E402

CLASS_COLORS = {'divider': 'orange', 'ped_crossing': 'tab:blue', 'boundary': 'red', 'centerline': 'tab:green'}


def load_extras(path, epsg):
    """Stop lines, traffic lights and signs (ways with a type tag) in the map CRS."""
    from pyproj import Transformer
    root = ET.parse(path).getroot()
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    nodes = {n.get('id'): tr.transform(float(n.get('lon')), float(n.get('lat'))) for n in root.iter('node')}
    extras = {}
    for w in root.iter('way'):
        tags = {t.get('k'): t.get('v') for t in w.iter('tag')}
        if tags.get('type') in ('stop_line', 'traffic_light', 'traffic_sign'):
            extras.setdefault(tags['type'], []).append(np.array([nodes[nd.get('ref')] for nd in w.iter('nd')]))
    return extras


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('osm')
    ap.add_argument('-o', '--out', help='PNG path (default: <map>_lanelet2.png next to the map)')
    ap.add_argument('--run', help='a run directory of run_mcap_pipeline.sh: draw its ego path')
    ap.add_argument('--labels', action='store_true', help='write lanelet names / ids')
    ap.add_argument('--show', action='store_true', help='open an interactive window instead of only saving')
    args = ap.parse_args()
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lanelets, epsg = load_lanelet2(args.osm)
    gt = WorldGT(lanelets, map_buffer=3.0)
    extras = load_extras(args.osm, epsg)
    allpts = np.concatenate([np.vstack([ll['left'], ll['right']]) for ll in lanelets])
    o = allpts.min(0)

    fig, axes = plt.subplots(1, 2, figsize=(30, 17), sharex=True, sharey=True)
    ax = axes[0]
    for ll in lanelets:
        poly = np.vstack([ll['left'], ll['right'][::-1]]) - o
        cross = ll['subtype'] == 'crosswalk'
        ax.fill(*poly.T, color='tab:blue' if cross else '0.8', alpha=0.6 if cross else 0.5, lw=0)
        for b in (ll['left'], ll['right']):
            ax.plot(*(b - o).T, color='0.35', lw=0.5)
        if args.labels and not cross:
            c = (ll['left'].mean(0) + ll['right'].mean(0)) / 2 - o
            ax.text(*c, ll['name'] or ll['id'], fontsize=4)
    for line in extras.get('stop_line', []):
        ax.plot(*(line - o).T, color='red', lw=2)
    for kind, color, marker in (('traffic_light', 'gold', 'o'), ('traffic_sign', 'magenta', 's')):
        pts = [l.mean(0) - o for l in extras.get(kind, [])]
        if pts:
            ax.scatter(*np.array(pts).T, s=25, c=color, marker=marker, edgecolors='k', linewidths=0.4, zorder=5)
    n = {k: len(v) for k, v in extras.items()}
    ax.set_title(f'{Path(args.osm).name}: {gt.n_road} road + {gt.n_cross} crosswalk lanelets, '
                 f'{n.get("stop_line", 0)} stop lines, {n.get("traffic_light", 0)} traffic lights, '
                 f'{n.get("traffic_sign", 0)} signs  (EPSG:{epsg})')

    ax = axes[1]
    for poly in getattr(gt.junctions, 'geoms', [gt.junctions]):
        if not poly.is_empty:
            ax.fill(*(np.array(poly.exterior.coords) - o).T, color='0.85', lw=0)
    lines = gt.world_lines()
    for c in MAP_CLASSES:
        for i, l in enumerate(lines[c]):
            ax.plot(*(l - o).T, color=CLASS_COLORS[c], lw=0.8 if c == 'centerline' else 1.2,
                    label=f'{c} ({len(lines[c])})' if i == 0 else None)
    ax.fill([], [], color='0.85', label='junction (no dividers)')
    for j in find_junctions(gt):  # same names as eval/junctions/J<n>.png
        for a in axes:
            a.annotate(f'{j["name"]} ({j["compass"]})', j['center'] - o, xytext=(12, 12), textcoords='offset points',
                       fontsize=11, weight='bold', arrowprops=dict(arrowstyle='-', lw=0.8))
    ax.set_title('SemVecNet ground truth derived from it (tools/mcap_eval_lanelet2.py)')
    ax.legend(loc='upper right')

    if args.run:
        run = Path(args.run)
        meta = json.loads((run / 'meta.json').read_text())
        poses = np.array([json.loads(l)['pose'][:2] for l in (run / 'frames.jsonl').read_text().splitlines() if l.strip()])
        path = poses + np.array(meta['origin'][:2]) - o
        for a in axes:
            a.plot(*path.T, color='k', lw=1, label=f'ego path ({run.name})')
            a.plot(*path[0], 'k^', ms=8)
        axes[0].legend(loc='upper right')

    for a in axes:
        a.set_aspect('equal')
        a.set_xlabel('east [m]')
        a.grid(alpha=0.3)
    axes[0].set_ylabel('north [m]')
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.osm).with_name(Path(args.osm).stem + '_lanelet2.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Wrote {out}')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
