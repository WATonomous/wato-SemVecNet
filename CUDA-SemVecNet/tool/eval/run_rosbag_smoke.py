#!/usr/bin/env python3
"""Run the eight-camera + LiDAR rosbag through the C++ SemVecNet pipeline."""

import argparse
from bisect import bisect_left
from collections import Counter
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


PROJECT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BAG = Path.home() / "Desktop/data/bags/may_30_ring_road_test_5"
CAMERAS = [f"/camera_pano_{d}" for d in ("ee", "ne", "nn", "nw", "se", "ss", "sw", "ww")]
LIDAR = "/lidar/all/points_merged"
ODOM = "/novatel/oem7/odom"
EGO = "rear_axle"
SECOND = 1_000_000_000
CAM_SLOP = 75_000_000
POSE_GAP = 100_000_000
FRAME_INTERVAL = 500_000_000
WINDOW = 6 * SECOND
OVERLAP = SECOND // 2


def stamp(msg):
    return msg.header.stamp.sec * SECOND + msg.header.stamp.nanosec


def nearest(samples, target):
    i = bisect_left(samples, (target,))
    return min(samples[max(0, i - 1):i + 1], key=lambda m: abs(m[0] - target), default=None)


def extract(bag, output, start, frame_count):
    # Only the reader container needs these dependencies, not the host Python.
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp

    def transform(t, q):
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        matrix[:3, 3] = [t.x, t.y, t.z]
        return matrix

    def point_features(msg):
        # Respect per-field offsets, byte order, and per-row padding in PointCloud2.
        fields = {f.name: f for f in msg.fields}
        names = ("x", "y", "z", "intensity")
        if any(n not in fields or fields[n].datatype != 7 or fields[n].count != 1 for n in names):
            raise ValueError("Expected scalar float32 x/y/z/intensity in merged PointCloud2")
        dtype = np.dtype({
            "names": names, "formats": [">f4" if msg.is_bigendian else "<f4"] * 4,
            "offsets": [fields[n].offset for n in names], "itemsize": msg.point_step,
        })
        cloud = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=bytes(msg.data),
                           strides=(msg.row_step, msg.point_step))
        features = np.zeros((cloud.size, 5), dtype="<f4")
        for i, name in enumerate(names):
            features[:, i] = cloud[name].ravel()
        # The merged cloud has no per-point time field; use zero time lag.
        keep = (np.isfinite(features).all(axis=1)
                & (np.abs(features) <= np.finfo(np.float16).max).all(axis=1)
                & (np.abs(features[:, :3]).sum(axis=1) > 0.1))
        return features[keep].astype("<f2")

    image_topics = [f"{c}/image_rect_compressed" for c in CAMERAS]
    info_topics = [f"{c}/camera_info" for c in CAMERAS]
    topics = image_topics + info_topics + [LIDAR, ODOM]
    mounts = {}

    def windows(stream):
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        summary = reader.get_summary()
        if not summary or not summary.statistics or not summary.chunk_indexes:
            raise ValueError("Expected an indexed ROS 2 MCAP with summary statistics")
        missing = set(topics + ["/tf_static"]) - {c.topic for c in summary.channels.values()}
        if missing:
            raise ValueError(f"Bag is missing required topics: {', '.join(sorted(missing))}")
        t0 = summary.statistics.message_start_time
        # Fetch the rig mounts separately so --start can seek past startup.
        for record in reader.iter_decoded_messages(topics=["/tf_static"], ending_before=t0 + 3 * SECOND):
            for tf in record.decoded_message.transforms:
                mounts[tf.child_frame_id.lstrip("/")] = (
                    tf.header.frame_id.lstrip("/"),
                    transform(tf.transform.translation, tf.transform.rotation),
                )
        end = summary.statistics.message_end_time + 1
        for lo in range(t0 + round(start * SECOND), end, WINDOW):
            hi = min(end, lo + WINDOW)
            records = {topic: [] for topic in topics}
            # Neighboring camera and odometry messages are context only; each
            # LiDAR message belongs to exactly one window by MCAP log time.
            for record in reader.iter_decoded_messages(
                topics=topics, starting_at=max(t0, lo - OVERLAP), ending_before=min(end, hi + OVERLAP),
            ):
                topic = record.channel.topic
                if topic != LIDAR or lo <= record.message.log_time < hi:
                    msg = record.decoded_message
                    records[topic].append((stamp(msg), msg))
            for samples in records.values():
                samples.sort(key=lambda m: m[0])
            yield t0, records

    def root_pose(frame):
        pose = np.eye(4)
        visited = set()
        while frame in mounts:
            if frame in visited:
                raise ValueError(f"Cycle in /tf_static at {frame}")
            visited.add(frame)
            frame, link = mounts[frame]
            pose = link @ pose
        return frame, pose

    def lookup(target_frame, source_frame):
        target_root, root_from_target = root_pose(target_frame)
        source_root, root_from_source = root_pose(source_frame)
        if target_root != source_root:
            raise ValueError(f"/tf_static does not connect {source_frame} to {target_frame}")
        return np.linalg.solve(root_from_target, root_from_source)

    def process_window(t0, records):
        odom = sorted(dict(records[ODOM]).items())
        frames = {(m.header.frame_id, m.child_frame_id) for _, m in odom}
        if len(odom) < 2:
            skipped["odometry"] += len(records[LIDAR])
            return
        if len(frames) != 1:
            raise ValueError("Expected one consistent odometry frame pair")
        _, base_frame = next(iter(frames))
        base_frame = base_frame.lstrip("/")
        base_from_ego = lookup(base_frame, EGO)
        pose_times = [t for t, _ in odom]
        poses = np.stack([transform(m.pose.pose.position, m.pose.pose.orientation) for _, m in odom])
        rotations = Slerp([(t - t0) / SECOND for t in pose_times], Rotation.from_matrix(poses[:, :3, :3]))

        def pose_at(t):
            i = bisect_left(pose_times, t)
            if i < len(pose_times) and pose_times[i] == t:
                return poses[i]
            if not 0 < i < len(pose_times) or pose_times[i] - pose_times[i - 1] > POSE_GAP:
                return None
            alpha = (t - pose_times[i - 1]) / (pose_times[i] - pose_times[i - 1])
            pose = np.eye(4)
            pose[:3, :3] = rotations((t - t0) / SECOND).as_matrix()
            pose[:3, 3] = (1 - alpha) * poses[i - 1, :3, 3] + alpha * poses[i, :3, 3]
            return pose

        nonlocal origin, saved, last
        for t, cloud in records[LIDAR]:
            if frame_count and (t < t0 + round(start * SECOND) or t < last + FRAME_INTERVAL):
                continue
            world_from_base = pose_at(t)
            if world_from_base is None:
                skipped["odometry"] += 1
                continue
            images = [nearest(records[topic], t) for topic in image_topics]
            if any(pair is None or abs(pair[0] - t) > CAM_SLOP for pair in images):
                skipped["camera sync"] += 1
                continue
            infos = [nearest(records[topic], pair[0]) for topic, pair in zip(info_topics, images)]
            if any(info is None or info[0] != pair[0] for info, pair in zip(infos, images)):
                skipped["CameraInfo sync"] += 1
                continue
            camera_poses = [pose_at(s) for s, _ in images]
            if any(pose is None for pose in camera_poses):
                skipped["odometry"] += 1
                continue
            points = point_features(cloud)
            if len(points) == 0:
                skipped["empty cloud"] += 1
                continue
            if t < last:
                raise ValueError("LiDAR timestamps went backwards across MCAP windows")
            emit(t, cloud, world_from_base, base_frame, base_from_ego, images, infos, camera_poses, points)
            saved += 1
            last = t
            if frame_count and saved == frame_count:
                return

    def emit(t, cloud, world_from_base, base_frame, base_from_ego, images, infos, camera_poses, points):
        nonlocal origin

        dims = {(info.width, info.height) for _, info in infos}
        if len(dims) != 1:
            raise ValueError("C++ forward expects all eight cameras to have the same dimensions")
        width, height = next(iter(dims))
        if width <= 0 or height <= 0:
            raise ValueError("Invalid CameraInfo dimensions")
        world_from_lidar = world_from_base @ lookup(base_frame, cloud.header.frame_id.lstrip("/"))
        world_from_ego = world_from_base @ base_from_ego
        if origin is None:
            origin = world_from_ego[:3, 3].copy()
        # Rebase translations in double precision before handing poses to the float32 mapper.
        lidar_to_map, ego_to_map = world_from_lidar.copy(), world_from_ego.copy()
        lidar_to_map[:3, 3] -= origin
        ego_to_map[:3, 3] -= origin
        output.write(struct.pack("<QIIII", t, len(CAMERAS), width, height, len(points)))
        output.write(np.stack([lidar_to_map, ego_to_map]).astype("<f4").tobytes())
        for (_, image), (_, info), pose in zip(images, infos, camera_poses):
            frame = image.header.frame_id
            if frame != info.header.frame_id:
                raise ValueError(f"Image and CameraInfo frame IDs disagree: {frame} vs {info.header.frame_id}")
            # The images are rectified: P[:, :3] is the intrinsic matrix.
            k = np.array(info.p).reshape(3, 4)[:, :3].copy()
            if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
                raise ValueError(f"Missing rectified intrinsics for {frame}")
            world_from_camera = pose @ lookup(base_frame, frame.lstrip("/"))
            lidar_to_camera = np.linalg.solve(world_from_camera, world_from_lidar)
            payload = bytes(image.data)
            output.write(k.astype("<f4").tobytes())
            output.write(lidar_to_camera.astype("<f4").tobytes())
            output.write(struct.pack("<I", len(payload)))
            output.write(payload)
        output.write(points.tobytes())
        output.flush()

    origin = None
    saved, last, seen = 0, -SECOND, 0
    skipped = Counter()
    with bag.open("rb") as stream:
        for t0, records in windows(stream):
            seen += len(records[LIDAR])
            process_window(t0, records)
            if frame_count and saved == frame_count:
                break
    if not saved or (frame_count and saved != frame_count):
        raise ValueError(f"Found only {saved} synchronized frames; skipped {dict(skipped)}")
    if not frame_count:
        print(f"MCAP: {seen} LiDAR sweeps, {saved} synchronized, {sum(skipped.values())} skipped "
              f"({dict(skipped)})", file=sys.stderr, flush=True)


def run(args):
    bag = args.bag.expanduser().resolve()
    if bag.is_dir():
        files = sorted(bag.glob("*.mcap"))
        if len(files) != 1:
            raise ValueError(f"Expected one MCAP in {bag}; pass a specific .mcap file")
        bag = files[0].resolve()
    plan, model = (p.expanduser().resolve() for p in (args.hrnet, args.onnx))
    for path in (bag, plan, model):
        if not path.is_file():
            raise ValueError(f"Missing input file: {path}")

    image = os.environ.get("SEMVECNET_IMAGE", "semvecnet:dev-ort")
    reader_image = os.environ.get("SEMVECNET_MCAP_IMAGE", "semvecnet-mcap-hrnet:latest")
    if subprocess.run(["docker", "image", "inspect", reader_image], stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode:
        raise ValueError(f"Missing reader image {reader_image}; set SEMVECNET_MCAP_IMAGE to a local "
                         "image with Python, numpy, scipy, mcap>=1.6 and mcap-ros2-support")

    docker = ["docker", "run", "--rm", "--network", "none",
              "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
    subprocess.run([str(PROJECT / "tool/build.sh")], check=True,
                   env={**os.environ, "SEMVECNET_QUIET": "1"})
    if args.profile:
        out = Path("/tmp/semvecnet-profile-unused")
    elif args.out:
        out = args.out.expanduser().resolve()
        out.mkdir(parents=True, exist_ok=False)
    else:
        out = Path(tempfile.mkdtemp(prefix="semvecnet-rosbag-", dir=Path.cwd()))
    if args.profile:
        print("Profiling only; inference outputs will not be written.", flush=True)
    else:
        print(f"Writing results to {out}", flush=True)
    extract_cmd = [
        *docker, "--log-driver", "none",
        "-v", f"{Path(__file__).resolve()}:/extract.py:ro", "-v", f"{bag}:/bag.mcap:ro",
        "--entrypoint", "python3", reader_image, "/extract.py", "--extract",
        "/bag.mcap", str(args.start), str(args.frames),
    ]
    infer_cmd = [
        *docker, "-i", "--gpus", "all", "--entrypoint", "",
        "-v", f"{PROJECT}:{PROJECT}:ro",
        "-v", f"{plan}:/hrnet.plan:ro", "-v", f"{model}:/semvecnet.onnx:ro",
        image, str(PROJECT / "build/semvecnet"),
        "/hrnet.plan", "/semvecnet.onnx", "/output",
    ]
    if not args.profile:
        infer_cmd[ infer_cmd.index(image):infer_cmd.index(image) ] = ["-v", f"{out}:/output"]
    if args.profile:
        infer_cmd.append("--profile")
    with subprocess.Popen(extract_cmd, stdout=subprocess.PIPE) as extractor:
        try:
            with subprocess.Popen(infer_cmd, stdin=extractor.stdout) as inference:
                extractor.stdout.close()
                status = inference.wait()
        finally:
            extractor.stdout.close()
        if status:
            suffix = "" if args.profile else f"; partial output: {out}"
            raise ValueError(f"C++ inference exited with status {status}{suffix}")
        if extractor.wait():
            raise ValueError(f"MCAP extraction failed; partial output: {out}")


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--extract":
        extract(Path(sys.argv[2]), sys.stdout.buffer, float(sys.argv[3]), int(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, nargs="?", default=DEFAULT_BAG,
                        help="MCAP file or directory containing a single MCAP (default: the May 30 test 5 bag)")
    parser.add_argument("--frames", type=int, default=2,
                        help="sample count at 0.5 s intervals, or 0 to process every synchronized sweep (default: 2)")
    parser.add_argument("--start", type=float, help="seconds from bag start (default: 0 for all, 0.5 for a sample)")
    parser.add_argument("--out", type=Path, help="new output directory (default: create one in the current directory)")
    parser.add_argument("--profile", action="store_true",
                        help="report per-stage timings without writing inference outputs")
    parser.add_argument("--hrnet", type=Path, default=PROJECT / "model/hrnet_ocrnet_mapillary_b8_s1.plan")
    parser.add_argument("--onnx", type=Path, default=PROJECT / "models/semvecnet.onnx")
    args = parser.parse_args()
    if args.profile and args.out:
        parser.error("--out cannot be used with --profile")
    if args.start is None:
        args.start = 0.5 if args.frames else 0
    if args.frames < 0 or not math.isfinite(args.start) or args.start < 0:
        parser.error("expected nonnegative --frames and finite, nonnegative --start")
    run(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"Rosbag smoke test: {error}")
