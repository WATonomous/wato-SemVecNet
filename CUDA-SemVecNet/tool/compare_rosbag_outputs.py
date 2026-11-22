#!/usr/bin/env python3
"""Compare timestamp-matched rosbag outputs and rerun Python on identical C++ maps."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def load_image(path):
    image = cv2.imread(str(path))
    if image is None or image.shape != (300, 150, 3):
        raise ValueError(f"Expected a 150 x 300 semantic map: {path}")
    return image


def pair(path, title, left, right, left_label, right_label):
    height, width = left.shape[:2]
    image = np.full((height + 62, 2 * width + 2, 3), 24, dtype=np.uint8)
    image[62:, :width] = left
    image[62:, width + 2:] = right
    for text, x, y in [(title, 8, 21), (left_label, 8, 49), (right_label, width + 10, 49)]:
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Could not write {path}")


def compare_vectors(py_vectors, cpp_vectors):
    """Match lines by class and ordered point distance, without relying on score sort order."""
    errors, scores = [], []
    for cls in sorted({v["cls"] for v in py_vectors + cpp_vectors}):
        py = [v for v in py_vectors if v["cls"] == cls]
        cpp = [v for v in cpp_vectors if v["cls"] == cls]
        if not py or not cpp:
            continue
        py_pts = np.array([v["pts"] for v in py])
        cpp_pts = np.array([v["pts"] for v in cpp])
        distances = np.linalg.norm(py_pts[:, None] - cpp_pts[None, :], axis=-1)
        ii, jj = linear_sum_assignment(distances.mean(axis=-1))
        errors.extend(distances[ii, jj])
        scores.extend(abs(py[i]["score"] - cpp[j]["score"]) for i, j in zip(ii, jj))
    return {
        "python_vectors": len(py_vectors), "cpp_vectors": len(cpp_vectors), "matched": len(errors),
        "mean_point_error_m": float(np.mean(errors)) if errors else None,
        "max_point_error_m": float(np.max(errors)) if errors else None,
        "max_score_error": max(scores, default=None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-repo", type=Path, required=True)
    parser.add_argument("--python-run", type=Path, required=True)
    parser.add_argument("--cpp-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parity-frames", type=int, default=5,
                        help="evenly spaced matched frames to rerun in PyTorch on the C++ PNG (0 disables)")
    args = parser.parse_args()
    if args.parity_frames < 0:
        parser.error("--parity-frames must be nonnegative")

    sys.path.insert(0, str(args.python_repo / "tools"))
    from mcap_compare_viz import CLASS_BGR, MAP_CLASSES, PH, PW, blank, ego_marker, ego_to_px

    def vector_panel(vectors):
        panel = blank()
        for line in vectors:
            px = ego_to_px(line["pts"]).round().astype(np.int32)
            color = CLASS_BGR[MAP_CLASSES.index(line["cls"])]
            cv2.polylines(panel, [px], False, color, 2, cv2.LINE_AA)
            cv2.circle(panel, tuple(px[-1]), 3, color, -1)
        return ego_marker(panel)

    cpp_by_stamp = {}
    paths = sorted((args.cpp_results / "polylines").glob("*.json"))
    # The C++ run writes JSON first and the combined PNG last.
    for path in paths:
        if (args.cpp_results / "combined" / f"{path.stem}.png").exists():
            record = json.loads(path.read_text())
            cpp_by_stamp[record["stamp_ns"]] = (path.stem, record)
    matched = [(frame, *cpp_by_stamp[frame["stamp_ns"]])
               for line in (args.python_run / "frames.jsonl").read_text().splitlines()
               if (frame := json.loads(line))["stamp_ns"] in cpp_by_stamp]
    if not matched:
        raise ValueError("No frames matched by exact LiDAR timestamp")
    meta = json.loads((args.python_run / "predictions.json").read_text())["meta"]
    semantic_dir = args.out / "semantic_maps"
    vector_dir = args.out / "polylines"
    parity_dir = args.out / "same_input"
    semantic_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(exist_ok=True)
    report = {"matched_frames": len(matched), "python_checkpoint": meta["checkpoint"],
              "end_to_end": [], "same_input": []}
    for frame, cpp_key, cpp in matched:
        key = f'{frame["idx"]:06d}'
        py_map = load_image(args.python_run / frame["map"])
        cpp_map = load_image(args.cpp_results / "semantic_maps" / f"{cpp_key}.png")
        py = json.loads((args.python_run / "vectors" / f"{key}.json").read_text())
        if py["stamp_ns"] != cpp["stamp_ns"]:
            raise ValueError(f"Vector timestamps do not match for {key}")
        title = f'LiDAR timestamp {cpp["stamp_ns"]}'
        py_label, cpp_label = f"Python {key}", f"C++ {cpp_key}"
        pair(semantic_dir / f"{key}.png", "Semantic maps | " + title,
             cv2.resize(py_map, (PW, PH), interpolation=cv2.INTER_NEAREST),
             cv2.resize(cpp_map, (PW, PH), interpolation=cv2.INTER_NEAREST), py_label, cpp_label)
        pair(vector_dir / f"{key}.png", "Predicted polylines | " + title,
             vector_panel(py["vectors"]), vector_panel(cpp["vectors"]),
             f'{py_label} ({len(py["vectors"])} lines)', f'{cpp_label} ({len(cpp["vectors"])} lines)')
        different = np.any(py_map != cpp_map, axis=2)
        observed = np.any(py_map != 0, axis=2) | np.any(cpp_map != 0, axis=2)
        report["end_to_end"].append({
            "python_frame": key, "cpp_frame": cpp_key, "stamp_ns": cpp["stamp_ns"],
            "different_pixel_fraction": float(different.mean()),
            "different_observed_pixel_fraction": float(different.sum() / max(observed.sum(), 1)),
            "python_vectors": len(py["vectors"]), "cpp_vectors": len(cpp["vectors"]),
        })
    print(f"Saved {len(matched)} timestamp-matched semantic map and polyline pairs", flush=True)

    if args.parity_frames:
        import torch
        from mcap_infer_semvecnet import build_model, load_map, predict, to_ego

        torch.set_num_threads(4)  # Keep the GPU available to the full-bag C++ run.
        source = args.python_repo / "SemVecNet"
        checkpoint = source / meta["checkpoint"]
        cfg = source / "projects/configs/semvecnet/semvecnet_nusc_centerline.py"
        model, norm = build_model(str(cfg), str(checkpoint), "cpu")
        indices = np.linspace(0, len(matched) - 1, min(args.parity_frames, len(matched)), dtype=int)
        parity_dir.mkdir(exist_ok=True)
        for idx in indices:
            frame, cpp_key, cpp = matched[idx]
            model_input = load_map(args.cpp_results / "semantic_maps" / f"{cpp_key}.png", norm)
            (pts, scores, labels), = predict(model, [model_input], "cpu")
            py_vectors = [{"cls": MAP_CLASSES[int(label)], "score": float(score), "pts": to_ego(p).tolist()}
                          for p, score, label in zip(pts, scores, labels) if float(score) > 0.4]
            errors = compare_vectors(py_vectors, cpp["vectors"])
            report["same_input"].append({"cpp_frame": cpp_key, **errors})
            pair(parity_dir / f"{cpp_key}.png", f"Same C++ semantic map | {cpp_key}",
                 vector_panel(py_vectors), vector_panel(cpp["vectors"]),
                 f"Python PyTorch ({len(py_vectors)} lines)", f'C++ ONNX ({len(cpp["vectors"])} lines)')
            print(f"{cpp_key}: {errors}", flush=True)
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
