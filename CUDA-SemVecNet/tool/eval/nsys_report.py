#!/usr/bin/env python3
"""Turn an Nsight Systems run of build/semvecnet into a readable timing report.

Reads the CSVs that `nsys stats` exports for the NVTX ranges in
SemVecNet::forward() and its stage implementations, and for CUDA kernels,
then writes:

  report.html          per-stage breakdown, per-frame timeline, top kernels
  stage_timings.csv    one row per frame, one column per stage (ms)

and prints a short summary to stdout. Standard library only (Python >= 3.8).
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import sys
from pathlib import Path

PREFIX = "SemVecNet::"
TOTAL = PREFIX + "forward"

# Pipeline order, as called in SemVecNet::forward(): key, label, description.
STAGES = [
    ("camera_preprocess", "Camera preprocess", "Resize + normalize the RGB images for HRNet"),
    ("lidar_preprocess", "LiDAR preprocess", "FP16 points to FP32 x, y, z, intensity, time lag"),
    ("hrnet_inference", "HRNet inference", "TensorRT HRNet-OCRNet: class mask per camera"),
    ("semantic_projection", "Semantic projection", "CUDA: label LiDAR points from the masks"),
    ("bev_mapping", "BEV mapping", "Paint labeled points into the global grid"),
    ("map_render", "Map render", "Crop + rotate the ego-centric model input"),
    ("semvecnet_model", "SemVecNet model", "ONNX Runtime: polyline points + class logits"),
    ("postprocess", "Postprocess", "Score, filter, convert polylines to meters"),
]
OTHER = ("other", "Other / overhead", "Time in forward() outside the named stages")

# Categorical slots 1-8 (light, dark), fixed order, one per stage. "Other" is neutral gray.
COLORS = [
    ("#2a78d6", "#3987e5"),
    ("#eb6834", "#d95926"),
    ("#1baf7a", "#199e70"),
    ("#eda100", "#c98500"),
    ("#e87ba4", "#d55181"),
    ("#008300", "#0ca30c"),
    ("#6250d6", "#9085e9"),
    ("#e34948", "#e66767"),
]
OTHER_COLOR = ("#a3a29c", "#6b6a64")


def read_csv(path: Path | None) -> list[dict]:
    if not path or not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ms(ns) -> float:
    return float(ns) / 1e6


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def load_frames(nvtx_csv: Path) -> list[dict]:
    """One dict per forward() call: {'total': ms, '<stage key>': ms, ...}."""
    rows = []
    for raw in read_csv(nvtx_csv):
        r = {k.replace(" (ns)", ""): v for k, v in raw.items()}  # "Start (ns)" -> "Start"
        # nsys shows default-domain ranges as ":Name"; drop the empty domain.
        name = r.get("Name", "")
        name = name[1:] if name.startswith(":") else name
        if name.startswith(PREFIX):
            r["Name"] = name
            rows.append(r)
    totals = sorted((r for r in rows if r["Name"] == TOTAL), key=lambda r: int(r["Start"]))
    by_parent: dict[str, list[dict]] = {}
    for r in rows:
        by_parent.setdefault(r.get("ParentId", ""), []).append(r)

    frames = []
    for t in totals:
        frame = {"total": ms(t["Duration"])}
        for child in by_parent.get(t["RangeId"], []):
            key = child["Name"][len(PREFIX):]
            frame[key] = frame.get(key, 0.0) + ms(child["Duration"])
        named = sum(frame.get(key, 0.0) for key, _, _ in STAGES)
        frame[OTHER[0]] = max(frame["total"] - named, 0.0)
        frames.append(frame)
    return frames


def load_kernels(kern_csv: Path, top: int) -> list[dict]:
    kernels = []
    for r in read_csv(kern_csv):
        try:
            kernels.append({
                "name": r["Name"],
                "share": float(r["Time (%)"]),
                "total": ms(r["Total Time (ns)"]),
                "count": int(r["Instances"]),
                "avg": ms(r["Avg (ns)"]),
                "max": ms(r["Max (ns)"]),
            })
        except (KeyError, ValueError):
            continue
    kernels.sort(key=lambda k: k["total"], reverse=True)
    return kernels[:top]


def summarize(frames: list[dict]) -> dict:
    keys = ["total"] + [k for k, _, _ in STAGES] + [OTHER[0]]
    summary = {}
    for key in keys:
        values = [f.get(key, 0.0) for f in frames]
        summary[key] = {
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
            "p95": percentile(values, 0.95),
        }
    return summary


def share(summary: dict, key: str) -> float:
    total = summary["total"]["median"]
    return 100.0 * summary[key]["median"] / total if total else 0.0


def short_kernel(name: str, limit: int) -> str:
    # "void ns::foo<...>(...)" -> "ns::foo"
    base = name[5:] if name.startswith("void ") else name
    for sep in ("<", "("):
        idx = base.find(sep)
        if idx > 0:
            base = base[:idx]
    return base if len(base) <= limit else base[: limit - 1] + "…"


# --------------------------------------------------------------------------- terminal


def print_summary(steady: list[dict], warmup: dict | None, summary: dict, kernels: list[dict], meta: dict) -> None:
    total = summary["total"]
    print()
    print("=" * 80)
    print(" SemVecNet::forward() timing   %s" % meta.get("bag", ""))
    print("=" * 80)
    print(" %d frame(s) measured%s" % (len(steady), " (frame 0 = warm-up, excluded)" if warmup else ""))
    print()
    print(" END-TO-END, ONE INPUT:  %.2f ms median  (%.1f FPS)" % (total["median"], 1000.0 / total["median"]))
    print("                         mean %.2f  min %.2f  p95 %.2f  max %.2f ms"
          % (total["mean"], total["min"], total["p95"], total["max"]))
    if warmup:
        print("   warm-up frame 0:      %.2f ms (lazy CUDA / cuDNN / ORT init)" % warmup["total"])
    print()
    print(" %-20s %8s %8s %8s  %s" % ("Stage (call order)", "median", "mean", "max", "share of forward()"))
    print(" " + "-" * 78)
    for key, label, _ in STAGES + [OTHER]:
        s = summary[key]
        pct = share(summary, key)
        bar = "█" * int(round(pct * 0.3)) or ("▏" if s["median"] > 0 else "")
        print(" %-20s %8.2f %8.2f %8.2f  %5.1f%% %s" % (label, s["median"], s["mean"], s["max"], pct, bar))
    print(" " + "-" * 78)
    print(" %-20s %8.2f %8.2f %8.2f  ms" % ("forward() total", total["median"], total["mean"], total["max"]))
    top = max(STAGES, key=lambda s: summary[s[0]]["median"])
    print()
    print(" Bottleneck: %s, %.1f%% of each frame" % (top[1], share(summary, top[0])))
    if kernels:
        print()
        print(" Top CUDA kernels by GPU time (whole run, incl. warm-up):")
        for k in kernels[:5]:
            print("   %5.1f%%  %8.2f ms  x%-5d %s" % (k["share"], k["total"], k["count"], short_kernel(k["name"], 46)))
    print()


# --------------------------------------------------------------------------- csv


def write_csv(path: Path, frames: list[dict], warmup: dict | None) -> None:
    keys = [k for k, _, _ in STAGES] + [OTHER[0]]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "warmup", "forward_total_ms"] + [k + "_ms" for k in keys])
        all_frames = ([warmup] if warmup else []) + frames
        for i, fr in enumerate(all_frames):
            w.writerow([i, int(warmup is not None and i == 0), "%.4f" % fr["total"]] + ["%.4f" % fr.get(k, 0.0) for k in keys])


# --------------------------------------------------------------------------- html


def write_html(path: Path, steady: list[dict], warmup: dict | None, summary: dict, kernels: list[dict], meta: dict) -> None:
    total = summary["total"]
    entries = STAGES + [OTHER]
    colors = [(key, *(COLORS[i] if i < len(COLORS) else OTHER_COLOR)) for i, (key, _, _) in enumerate(entries)]
    light_css = "\n".join("  --c-%s: %s;" % (k, light) for k, light, _ in colors)
    dark_css = "\n".join("    --c-%s: %s;" % (k, dark) for k, _, dark in colors)
    esc = html.escape

    bottleneck = max(STAGES, key=lambda s: summary[s[0]]["median"])

    # Stage breakdown: horizontal bars ranked by median ms, colored by stage identity.
    max_stage = max(summary[k]["median"] for k, _, _ in entries) or 1.0
    bars = []
    for key, label, desc in sorted(entries, key=lambda e: summary[e[0]]["median"], reverse=True):
        s = summary[key]
        bars.append(
            '<div class="row" tabindex="0" data-tip="%s|median %.2f ms, mean %.2f, p95 %.2f, max %.2f|%.1f%% of forward()">'
            '<div class="lbl"><span class="sw" style="background:var(--c-%s)"></span>%s<small>%s</small></div>'
            '<div class="track"><div class="bar" style="width:%.3f%%;background:var(--c-%s)"></div></div>'
            '<div class="val"><b>%.2f</b> ms<span>%.1f%%</span></div></div>'
            % (esc(label), s["median"], s["mean"], s["p95"], s["max"], share(summary, key),
               key, esc(label), esc(desc), 100 * s["median"] / max_stage, key, s["median"], share(summary, key))
        )

    # Per-frame stacked bars, stages in pipeline order, all on one shared scale.
    labeled = ([("Warm-up", warmup)] if warmup else []) + [("Frame %d" % (i + (1 if warmup else 0)), f) for i, f in enumerate(steady)]
    scale_max = max(f["total"] for _, f in labeled) or 1.0
    frame_rows = []
    for name, fr in labeled:
        segs = "".join(
            '<div class="seg" style="width:%.4f%%;background:var(--c-%s)" data-tip="%s: %s|%.2f ms|%.1f%% of this frame"></div>'
            % (100 * fr.get(key, 0.0) / scale_max, key, esc(name), esc(label), fr.get(key, 0.0), 100 * fr.get(key, 0.0) / fr["total"])
            for key, label, _ in entries if fr.get(key, 0.0) > 0
        )
        frame_rows.append('<div class="frow%s"><div class="fname">%s</div><div class="stack">%s</div><div class="fval">%.1f ms</div></div>'
                          % (" warm" if name == "Warm-up" else "", name, segs, fr["total"]))
    legend = "".join('<span><i style="background:var(--c-%s)"></i>%s</span>' % (k, esc(l)) for k, l, _ in entries)

    table_rows = []
    for key, label, _ in entries + [("total", "forward() total", "")]:
        s = summary[key]
        sw = '<span class="sw" style="background:var(--c-%s)"></span>' % key if key != "total" else ""
        table_rows.append("<tr%s><td>%s%s</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.1f%%</td></tr>"
                          % (' class="tot"' if key == "total" else "", sw, esc(label),
                             s["median"], s["mean"], s["min"], s["p95"], s["max"], share(summary, key)))

    kernel_rows = "".join(
        '<tr><td title="%s"><code>%s</code></td><td>%d</td><td>%.2f</td><td>%.3f</td><td>%.3f</td><td>%.1f%%</td></tr>'
        % (esc(k["name"]), esc(short_kernel(k["name"], 70)), k["count"], k["total"], k["avg"], k["max"], k["share"])
        for k in kernels
    ) or '<tr><td colspan="6">No CUDA kernel data in this report.</td></tr>'

    warm_tile = ""
    if warmup:
        warm_tile = ('<div class="tile"><div class="k">Warm-up frame</div><div class="v">%.0f<span> ms</span></div>'
                     '<div class="s">First call (lazy CUDA / cuDNN / ORT init), excluded from stats</div></div>' % warmup["total"])

    doc = TEMPLATE.format(
        light_css=light_css,
        dark_css=dark_css,
        bag=esc(meta.get("bag", "")),
        start=esc(str(meta.get("start", ""))),
        created=esc(meta.get("created", "")),
        n=len(steady),
        median=total["median"],
        fps=1000.0 / total["median"] if total["median"] else 0,
        mean=total["mean"],
        p95=total["p95"],
        tmin=total["min"],
        tmax=total["max"],
        bn_label=esc(bottleneck[1]),
        bn_ms=summary[bottleneck[0]]["median"],
        bn_share=share(summary, bottleneck[0]),
        warm_tile=warm_tile,
        bars="".join(bars),
        frames="".join(frame_rows),
        legend=legend,
        table_rows="".join(table_rows),
        kernel_rows=kernel_rows,
        links="".join("<li><code>%s</code>: %s</li>" % (esc(p), esc(d)) for p, d in meta.get("files", [])),
        warm_note="The faded warm-up bar is excluded from every statistic." if warmup else "",
    )
    path.write_text(doc)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SemVecNet profile: {bag}</title>
<style>
:root {{
  color-scheme: light;
  --surface-0: #f4f4f2; --surface-1: #fcfcfb; --border: #e4e3df; --track: #efeeeb;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #6f6e69;
{light_css}
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    color-scheme: dark;
    --surface-0: #111110; --surface-1: #1a1a19; --border: #2c2c2a; --track: #242422;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
{dark_css}
  }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1080px; margin: 0 auto; padding: 32px 24px 64px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 16px; margin: 0 0 4px; }}
.meta, .sub {{ color: var(--text-secondary); margin: 0 0 16px; }}
.sub {{ font-size: 13px; }}
code {{ font: 12px/1.3 ui-monospace, "SF Mono", Menlo, monospace; }}
section {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 20px; margin-top: 16px; }}
.tiles {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 20px; }}
.tile {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }}
.tile.hero {{ grid-column: span 2; }}
.k {{ color: var(--text-secondary); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
.v {{ font-size: 32px; font-weight: 650; font-variant-numeric: tabular-nums; margin: 4px 0; }}
.hero .v {{ font-size: 44px; }}
.v span {{ font-size: 16px; font-weight: 500; color: var(--text-secondary); }}
.s {{ color: var(--text-muted); font-size: 12px; font-variant-numeric: tabular-nums; }}
.row {{ display: grid; grid-template-columns: 300px 1fr 130px; align-items: center; gap: 12px; padding: 6px 8px; border-radius: 6px; }}
.row:hover, .row:focus {{ background: var(--track); outline: none; }}
.lbl {{ display: flex; flex-direction: column; padding-left: 20px; position: relative; }}
.lbl .sw {{ position: absolute; left: 0; top: 4px; }}
.lbl small {{ color: var(--text-muted); font-size: 12px; }}
.sw {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 8px; vertical-align: -1px; }}
.track {{ height: 14px; }}
.bar {{ height: 100%; border-radius: 0 4px 4px 0; min-width: 2px; }}
.val {{ text-align: right; font-variant-numeric: tabular-nums; color: var(--text-secondary); }}
.val b {{ color: var(--text-primary); font-weight: 600; }}
.val span {{ display: inline-block; width: 56px; color: var(--text-muted); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 0 0 14px; font-size: 12px; color: var(--text-secondary); }}
.legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }}
.frow {{ display: grid; grid-template-columns: 80px 1fr 80px; align-items: center; gap: 12px; margin: 3px 0; }}
.fname {{ color: var(--text-secondary); font-size: 12px; }}
.frow.warm .fname {{ font-style: italic; }}
.frow.warm .stack {{ opacity: .5; }}
.fval {{ text-align: right; font-variant-numeric: tabular-nums; font-size: 12px; color: var(--text-secondary); }}
.stack {{ display: flex; height: 18px; gap: 2px; }}
.seg {{ flex: none; min-width: 1px; }}
.seg:first-child {{ border-radius: 4px 0 0 4px; }}
.seg:last-child {{ border-radius: 0 4px 4px 0; }}
.seg:hover {{ filter: brightness(1.15); }}
table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--border); }}
th {{ color: var(--text-secondary); font-weight: 500; font-size: 12px; }}
th:first-child, td:first-child {{ text-align: left; }}
tr.tot td {{ font-weight: 650; border-top: 2px solid var(--border); }}
td code {{ white-space: nowrap; }}
ul {{ margin: 8px 0 0; padding-left: 18px; color: var(--text-secondary); }}
#tip {{ position: fixed; pointer-events: none; background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,.15); display: none; z-index: 9; max-width: 340px; }}
#tip b {{ display: block; }} #tip span {{ display: block; color: var(--text-secondary); }}
@media (max-width: 760px) {{ .tiles {{ grid-template-columns: 1fr 1fr; }} .row {{ grid-template-columns: 1fr; }} }}
</style></head>
<body><main>
<h1>SemVecNet pipeline profile</h1>
<p class="meta">{bag} &middot; start {start}s &middot; {created}</p>

<div class="tiles">
  <div class="tile hero"><div class="k">End-to-end, one input (median forward())</div>
    <div class="v">{median:.2f}<span> ms &middot; {fps:.1f} FPS</span></div>
    <div class="s">mean {mean:.2f} &middot; min {tmin:.2f} &middot; p95 {p95:.2f} &middot; max {tmax:.2f} ms over {n} frame(s)</div></div>
  <div class="tile"><div class="k">Bottleneck</div><div class="v">{bn_share:.0f}<span>%</span></div>
    <div class="s">{bn_label}: {bn_ms:.2f} ms median</div></div>
  {warm_tile}
</div>

<section>
  <h2>Where the time goes</h2>
  <p class="sub">Median host elapsed time per stage of <code>SemVecNet::forward()</code>, largest first. These NVTX ranges include CPU work and any waiting, but asynchronous GPU work may continue after a range ends. Hover a row for mean, p95 and max.</p>
  {bars}
</section>

<section>
  <h2>Every frame</h2>
  <p class="sub">Each bar is one <code>forward()</code> call with host-side stage durations in pipeline order, all on one scale. Hover a segment for its time. {warm_note}</p>
  <div class="legend">{legend}</div>
  {frames}
</section>

<section>
  <h2>Stage statistics (ms)</h2>
  <table><thead><tr><th>Stage (call order)</th><th>Median</th><th>Mean</th><th>Min</th><th>p95</th><th>Max</th><th>Share</th></tr></thead>
  <tbody>{table_rows}</tbody></table>
</section>

<section>
  <h2>Top CUDA kernels</h2>
  <p class="sub">GPU time across the whole run, warm-up included. Hover a name for the full signature.</p>
  <table><thead><tr><th>Kernel</th><th>Calls</th><th>Total ms</th><th>Avg ms</th><th>Max ms</th><th>GPU share</th></tr></thead>
  <tbody>{kernel_rows}</tbody></table>
</section>

<section>
  <h2>Files</h2>
  <ul>{links}</ul>
</section>
</main>
<div id="tip"></div>
<script>
const tip = document.getElementById('tip');
function show(e) {{
  const el = e.target.closest && e.target.closest('[data-tip]');
  if (!el) {{ tip.style.display = 'none'; return; }}
  const parts = el.dataset.tip.split('|');
  tip.replaceChildren();
  parts.forEach((t, i) => {{ const n = document.createElement(i ? 'span' : 'b'); n.textContent = t; tip.appendChild(n); }});
  tip.style.display = 'block';
  const box = el.getBoundingClientRect(), r = tip.getBoundingClientRect();
  const x = e.type === 'mousemove' ? e.clientX : box.left, y = e.type === 'mousemove' ? e.clientY : box.bottom;
  tip.style.left = Math.max(8, Math.min(x + 14, innerWidth - r.width - 8)) + 'px';
  tip.style.top = Math.max(8, Math.min(y + 14, innerHeight - r.height - 8)) + 'px';
}}
document.addEventListener('mousemove', show);
document.addEventListener('focusin', show);
document.addEventListener('mouseleave', () => tip.style.display = 'none');
</script>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nvtx", type=Path, required=True, help="nsys stats nvtx_pushpop_trace CSV")
    ap.add_argument("--kernels", type=Path, help="nsys stats cuda_gpu_kern_sum CSV")
    ap.add_argument("--out", type=Path, required=True, help="directory for report.html and stage_timings.csv")
    ap.add_argument("--meta", default="{}", help='JSON: {"bag", "start", "created", "files": [[path, description], ...]}')
    ap.add_argument("--keep-warmup", action="store_true", help="include frame 0 in the statistics")
    ap.add_argument("--top-kernels", type=int, default=15)
    args = ap.parse_args()

    frames = load_frames(args.nvtx)
    if not frames:
        print("ERROR: no %s NVTX ranges in %s (is nvtx in nsys --trace?)" % (TOTAL, args.nvtx), file=sys.stderr)
        return 1
    warmup = None
    if not args.keep_warmup and len(frames) > 1:
        warmup, frames = frames[0], frames[1:]

    meta = json.loads(args.meta)
    summary = summarize(frames)
    kernels = load_kernels(args.kernels, args.top_kernels) if args.kernels else []

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "stage_timings.csv", frames, warmup)
    write_html(args.out / "report.html", frames, warmup, summary, kernels, meta)
    print_summary(frames, warmup, summary, kernels, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
