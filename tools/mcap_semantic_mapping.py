#!/usr/bin/env python3
"""Stage 1: semantic BEV maps from a ROS 2 MCAP bag, exactly as tools/nusc_semantic_mapping.py makes them for nuScenes.

Per 2 Hz keyframe (nuScenes sample rate): one LiDAR sweep (/lidar_cc/velodyne_points, the roof LiDAR like nuScenes'
LIDAR_TOP), the nearest image of every camera, each segmented by HRNet exactly as get_semantic_image() does, then
DynamicMap.mapping_nuscenes() labels the LiDAR points from the images, accumulates them in one map for the whole bag
and renders the ego-centric 60 x 30 m map. Only reading the bag is new: images + camera_info (P), /tf_static for the
sensor mounts, /novatel/oem7/odom for the ego pose (UTM).

Outputs in --out: maps/<idx>.png, cams/<idx>_all.jpg (the cameras used), cams/<idx>.jpg (front camera + labels),
frames.jsonl (idx, stamp_ns, map, pose = [x, y, yaw] of the ego relative to meta.json's origin), meta.json,
static_tf.json (sensor mounts, reused by later splits of the same recording, which have no /tf_static).
"""
import argparse
import json
import os
import os.path as osp
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation, Slerp

REPO = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, REPO)

PRESET_CAMERAS = dict(all=['nn', 'ne', 'nw', 'se', 'ss', 'sw'], pano8=['nn', 'ne', 'ee', 'se', 'ss', 'sw', 'ww', 'nw'])
LIDAR = '/lidar_cc/velodyne_points'
ODOM = '/novatel/oem7/odom'
PANO_GRID = [['nw', 'nn', 'ne'], ['ww', None, 'ee'], ['sw', 'ss', 'se']]


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def tf_matrix(t, q):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def read_pointcloud2(msg):
    """PointCloud2 -> (4, N) x, y, z, intensity, like LidarPointCloud.from_file(...).points."""
    kinds = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
    fields = {f.name: f for f in msg.fields}
    names = [n for n in ('x', 'y', 'z', 'intensity') if n in fields]
    dtype = np.dtype(dict(names=names, formats=['<' + kinds[fields[n].datatype] for n in names],
                          offsets=[fields[n].offset for n in names], itemsize=msg.point_step))
    arr = np.frombuffer(bytes(msg.data), dtype=dtype, count=msg.width * msg.height)
    pcd = np.zeros((5, len(arr)))
    for i, n in enumerate(('x', 'y', 'z', 'intensity')):
        if n in names:
            pcd[i] = arr[n]
    pcd = pcd[:, np.isfinite(pcd[:3]).all(0)]
    # row 4: when in the sweep each point was measured, 0..1. Our Velodynes leave their per-point `time` field at 0, so
    # it comes from the rotation: they spin clockwise seen from above (azimuth decreasing) from the first point stored
    # (the VLP-32C at 20 Hz and the VLS-128 at 10 Hz alike, whatever the cloud's row layout)
    if pcd.shape[1]:
        az = np.degrees(np.arctan2(pcd[1], pcd[0]))
        pcd[4] = ((az[0] - az) % 360.0) / 360.0
    return pcd


def deskew(pcd, stamp, sweep_ns, stamp_at, pose_at, T_base_lidar, bins=10):
    """Move every point to where it would be seen from the LiDAR at `stamp` (Argoverse 2 ships its sweeps like this;
    nuScenes triggers each camera as the LiDAR passes it). stamp_at: 'start' or 'end' - which end of the sweep the
    header stamp marks. Returns (4, N) x, y, z, intensity, or None without poses for the whole sweep."""
    T_ref = pose_at(stamp)
    if T_ref is None:
        return None
    T_ref_inv = np.linalg.inv(T_ref @ T_base_lidar)
    t = stamp + (pcd[4] - (1.0 if stamp_at == 'end' else 0.0)) * sweep_ns
    out = pcd[:4].copy()
    edges = np.linspace(t.min(), t.max() + 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (t >= a) & (t < b)
        T = pose_at((a + b) / 2)
        if T is None:
            return None
        M = T_ref_inv @ T @ T_base_lidar  # LiDAR at the point's time -> LiDAR at the stamp
        out[:3, sel] = M[:3, :3] @ pcd[:3, sel] + M[:3, 3:4]
    return out


class StaticTf:
    """/tf_static tree: T_parent_child edges, composed to any frame pair."""

    def __init__(self):
        self.edges = {}  # child -> (parent, T_parent_child)

    def to_root(self, frame):
        T = np.eye(4)
        while frame in self.edges:
            parent, E = self.edges[frame]
            T, frame = E @ T, parent
        return frame, T

    def lookup(self, target, source):
        """T_target_source."""
        (r1, Tt), (r2, Ts) = self.to_root(target), self.to_root(source)
        if r1 != r2:
            raise KeyError(f'no /tf_static path {target} <- {source}')
        return np.linalg.inv(Tt) @ Ts


def get_semantic_image(seg, image_in, seg_color_ref):
    """nusc_semantic_mapping.get_semantic_image(image_scale=0.5), unchanged."""
    import semantic_mapping.src.utils.mapillary_visualization as mapillary_visl
    image_in_resized = cv2.resize(image_in, (1920 // 2, 1440 // 2), interpolation=cv2.INTER_AREA)
    image_in_resized = seg.preprocess(image_in_resized)
    image_out = np.reshape(seg.segmentation(image_in_resized), (1, 1440 // 2, 1920 // 2)).astype(np.uint8).squeeze()
    colored = np.squeeze(mapillary_visl.apply_color_map(image_out, seg_color_ref)).astype(np.uint8)
    return cv2.resize(colored, (image_in.shape[1], image_in.shape[0]), interpolation=cv2.INTER_NEAREST)


def camera_mosaic(images, tile_w=224):
    """The cameras in their roof positions (front at top centre), for the dataset's check images."""
    tiles = {}
    for name, im in images.items():
        t = cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), (tile_w, int(tile_w * im.shape[0] / im.shape[1])))
        cv2.putText(t, name, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles[name] = t
    h = max(t.shape[0] for t in tiles.values())
    blank = np.zeros((h, tile_w, 3), np.uint8)
    pad = lambda t: cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    return np.vstack([np.hstack([pad(tiles.get(c, blank)) if c else blank for c in row]) for row in PANO_GRID])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bag', required=True, help='.mcap file')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cameras', default='', help='all, pano8, or a comma list (nn,ne or camera_pano_nn,...); default: every roof camera')
    ap.add_argument('--expect-cameras', type=int, help='fail if the bag does not have this many of the cameras')
    ap.add_argument('--hz', type=float, default=2.0, help='rate of the written maps (nuScenes samples are 2 Hz)')
    ap.add_argument('--pcd-range-max', type=float, default=30.0, help='only LiDAR points closer than this (m, '
                    'camera depth) take a label: 30 in the paper\'s own vehicle config (base_cfg.py), 100 in nuscenes_cfg.py')
    ap.add_argument('--lidar-stamp', choices=('end', 'start', 'none'), default='end', help='which end of the sweep '
                    'the LiDAR header stamp marks, for the per-point motion correction (none: no correction)')
    ap.add_argument('--sweep-ms', type=float, default=0.0, help='LiDAR revolution time, ms (0: measured from the '
                    'spacing of the LiDAR messages - 50 ms for the 20 Hz VLP-32C, 100 ms for the 10 Hz VLS-128)')
    ap.add_argument('--render-filter', type=int, default=1, help='1: render with DynamicMap.mapping()\'s 3x3 '
                    'smoothing of the class scores (the paper\'s live mapping); 0: unfiltered, as its nuScenes script')
    ap.add_argument('--sweep-hz', type=float, default=20.0, help='LiDAR sweeps mapped per second (the paper maps '
                    'every Argoverse 2 sweep, 10 Hz, with motion compensation: tools/av2_semantic_mapping.py)')
    ap.add_argument('--max-cam-dt', type=float, default=0.1, help='max |camera - LiDAR| stamp gap, s')
    ap.add_argument('--near-map', default='', help='Lanelet2 .osm: skip keyframes more than --near-map-margin from it')
    ap.add_argument('--near-map-margin', type=float, default=60.0)
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--camera-calibration', default='', help='camera mount corrections (tools/calibrate_cameras.py)')
    ap.add_argument('--engine', required=True)
    ap.add_argument('--seg-config', default=osp.join(REPO, 'semantic_mapping/config/config_65.json'))
    ap.add_argument('--confusion-matrix', default=osp.join(REPO, 'semantic_mapping/src/hrnet/assets/seg_weights/hrnet_cfn_1999.npy'))
    ap.add_argument('--dummy-image', default=osp.join(REPO, 'semantic_mapping/src/hrnet/assets/seg_weights/1682354962.572154522.jpg'))
    args = ap.parse_args()

    out = Path(args.out)
    for d in ('maps', 'cams'):
        (out / d).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ the paper's setup (nusc_semantic_mapping.main)
    os.chdir(REPO)  # the semantic_mapping modules read their configs relative to the repository root
    import semantic_mapping.src.utils.mapillary_visualization as mapillary_visl
    from semantic_mapping.src.dynamic_map import DynamicMap
    from semantic_mapping.src.hrnet.hrnet_semantic_segmentation_tensorrt import (
        HRNetSemanticSegmentationTensorRT, get_custom_hrnet_args)
    from semantic_mapping.src.node_config.nuscenes_cfg import get_cfg_defaults
    cfg = get_cfg_defaults()
    cfg.MAPPING.CONFUSION_MTX.LOAD_PATH = args.confusion_matrix
    cfg.VISION_SEM_SEG.DATASET_CONFIG = args.seg_config
    cfg.VISION_SEM_SEG.ENGINE_FILE = args.engine
    cfg.VISION_SEM_SEG.DUMMY_IMAGE_PATH = args.dummy_image
    cfg.MAPPING.PCD.RANGE_MAX = args.pcd_range_max
    seg_color_ref = mapillary_visl.get_labels(cfg.VISION_SEM_SEG.DATASET_CONFIG)
    seg = HRNetSemanticSegmentationTensorRT(get_custom_hrnet_args(cfg))
    dm = DynamicMap(cfg)  # one map for the bag (the paper keeps one per nuScenes location)

    # ------------------------------------------------ the bag
    with open(args.bag, 'rb') as fh:
        summary = make_reader(fh).get_summary()
    topics = {ch.topic for ch in summary.channels.values()}
    short = PRESET_CAMERAS.get(args.cameras) or [c.replace('camera_pano_', '') for c in args.cameras.split(',') if c] \
        or [c for row in PANO_GRID for c in row if c]  # default: every roof camera in the bag
    cams = [f'/camera_pano_{c}/image_rect_compressed' for c in short
            if f'/camera_pano_{c}/image_rect_compressed' in topics]
    print(f'Bag: {Path(args.bag).name}\nCameras ({len(cams)}): {", ".join(cams)}\nLiDAR: {LIDAR}   pose: {ODOM}', flush=True)
    if args.expect_cameras and len(cams) != args.expect_cameras:
        sys.exit(f'expected {args.expect_cameras} cameras, the bag has {len(cams)}')
    for t in (LIDAR, ODOM):
        if t not in topics:
            sys.exit(f'the bag has no {t}')
    infos = {c: c.rsplit('/', 1)[0] + '/camera_info' for c in cams}

    near_map = None
    if args.near_map:
        import xml.etree.ElementTree as ET
        from pyproj import Transformer
        from scipy.spatial import cKDTree
        ll = np.array([(float(n.get('lon')), float(n.get('lat'))) for n in ET.parse(args.near_map).getroot().iter('node')])
        zone = int((ll[:, 0].mean() + 180) // 6) + 1
        near_map = cKDTree(np.stack(Transformer.from_crs(4326, 32600 + zone, always_xy=True).transform(ll[:, 0], ll[:, 1]), 1))

    tf = StaticTf()
    static_file = out / 'static_tf.json'
    if static_file.is_file():  # sensor mounts saved by an earlier split of this recording
        for e in json.loads(static_file.read_text()):
            tf.edges[e['child']] = (e['parent'], np.array(e['T']))
    # camera mount corrections from tools/calibrate_cameras.py: camera frame -> 4x4 applied after the /tf_static mount
    calib = {k: np.array(v) for k, v in json.loads(Path(args.camera_calibration).read_text()).items()} \
        if args.camera_calibration else {}
    images = {c: deque(maxlen=30) for c in cams}  # (stamp, frame_id, jpeg)
    K = {}
    odom = deque(maxlen=400)  # (stamp, T_utm_base), in time order
    pending = deque()  # LiDAR keyframes waiting for later camera images
    frames, skipped = [], Counter()
    origin = None
    period, last_key = int(1e9 / args.hz), None
    sweep_period, last_sweep = int(1e9 / max(args.sweep_hz, args.hz)), None
    last_lidar, revolution_ns = None, 50e6  # LiDAR message spacing = one revolution (updated from the bag)
    tic = time.time()
    fjson = open(out / 'frames.jsonl', 'w')

    def ego_frame():
        return 'rear_axle' if 'rear_axle' in tf.edges else 'base_link'

    def pose_at(t):
        """T_utm_base at time t, interpolated between the two GNSS poses around it (None if they are > 0.1 s apart)."""
        after = next((i for i, (s, _) in enumerate(odom) if s >= t), None)
        if not after:
            return None
        (s0, T0), (s1, T1) = odom[after - 1], odom[after]
        if s1 - s0 > 1e8:
            return None
        w = min(max((t - s0) / max(s1 - s0, 1), 0.0), 1.0)  # GNSS messages can arrive slightly out of order
        T = np.eye(4)
        T[:3, 3] = (1 - w) * T0[:3, 3] + w * T1[:3, 3]
        T[:3, :3] = Slerp([0, 1], Rotation.from_matrix([T0[:3, :3], T1[:3, :3]]))([w]).as_matrix()[0]
        return T

    def process(stamp, lidar_frame, pcd, key, sweep_ns):
        if args.lidar_stamp == 'none':
            pcd = pcd[:4]
        else:
            try:
                pcd = deskew(pcd, stamp, sweep_ns, args.lidar_stamp, pose_at,
                             tf.lookup('base_link', lidar_frame))
            except KeyError:
                pcd = None
            if pcd is None:
                skipped['no ego pose'] += 1
                return
        nonlocal origin
        T_utm_base = pose_at(stamp)
        if T_utm_base is None:  # no GNSS pose around the sweep
            skipped['no ego pose'] += 1
            return
        try:
            T_base_ego, T_base_lidar = tf.lookup('base_link', ego_frame()), tf.lookup('base_link', lidar_frame)
        except KeyError:
            skipped['no /tf_static sensor mounts'] += 1
            return
        T_ego_to_map, T_lidar_to_map = T_utm_base @ T_base_ego, T_utm_base @ T_base_lidar
        if near_map is not None and near_map.query(T_ego_to_map[:2, 3])[0] > args.near_map_margin:
            skipped['off the --near-map map (keyframes)' if key else 'off the --near-map map (sweeps)'] += 1
            return
        if origin is None:  # UTM metres are too large for the map grid: start the map frame at the first pose
            origin = T_ego_to_map[:3, 3].copy()
        T_ego_to_map[:3, 3] -= origin
        T_lidar_to_map[:3, 3] -= origin

        semantic_images, intrinsics_, T_lidar_to_cameras, raw = {}, {}, {}, {}
        for c in cams:
            near = [m for m in images[c] if abs(m[0] - stamp) <= args.max_cam_dt * 1e9]
            if not near or c not in K:
                continue
            s_cam, frame_id, jpeg = min(near, key=lambda m: abs(m[0] - stamp))
            T_utm_base_cam = pose_at(s_cam)
            try:
                T_base_camera = tf.lookup('base_link', frame_id) @ calib.get(frame_id, np.eye(4))
            except KeyError:
                continue
            if T_utm_base_cam is None:
                continue
            # motion compensation as the paper's Argoverse 2 path (project_ego_to_img_motion_compensated): our cameras
            # are not triggered by the LiDAR, so a point is carried from the vehicle at the sweep's time to the
            # vehicle at the image's time before it is projected
            T_lidar_to_camera = (np.linalg.inv(T_base_camera) @ np.linalg.inv(T_utm_base_cam) @ T_utm_base
                                 @ T_base_lidar)
            image = cv2.cvtColor(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            semantic_images[c] = get_semantic_image(seg, image, seg_color_ref)
            intrinsics_[c], T_lidar_to_cameras[c], raw[c] = K[c], T_lidar_to_camera, image
        if len(semantic_images) < len(cams):
            skipped['camera image missing / too far in time'] += 1
            return

        _, map_rendered, _, _ = dm.mapping_nuscenes(pcd, semantic_images, list(semantic_images), intrinsics_,
                                                    T_lidar_to_cameras, T_lidar_to_map, T_ego_to_map)
        if not key:  # sweep only adds to the map, like every Argoverse 2 sweep in the paper
            return
        if args.render_filter:  # DynamicMap.mapping()'s rendering (3x3 smoothing of the class scores)
            map_rendered = dm.render_local_map(dm.map, T_ego_to_map, dm.render_ego_centric, filter=True)
        idx = len(frames)
        cv2.imwrite(str(out / 'maps' / f'{idx:06d}.png'), cv2.cvtColor(map_rendered.astype(np.uint8), cv2.COLOR_BGR2RGB))
        name = lambda c: c.split('/')[1].replace('camera_pano_', '')
        cv2.imwrite(str(out / 'cams' / f'{idx:06d}_seg.jpg'),  # what HRNet labelled in every camera
                    camera_mosaic({name(c): (0.4 * raw[c] + 0.6 * semantic_images[c]).astype(np.uint8) for c in raw}),
                    [cv2.IMWRITE_JPEG_QUALITY, 75])
        cv2.imwrite(str(out / 'cams' / f'{idx:06d}_all.jpg'), camera_mosaic({name(c): im for c, im in raw.items()}),
                    [cv2.IMWRITE_JPEG_QUALITY, 75])
        front = next((c for c in cams if '_nn/' in c), cams[0])
        cv2.imwrite(str(out / 'cams' / f'{idx:06d}.jpg'),
                    cv2.cvtColor((0.5 * raw[front] + 0.5 * semantic_images[front]).astype(np.uint8), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        rec = dict(idx=idx, stamp_ns=stamp, map=f'maps/{idx:06d}.png', cams=len(raw), front=f'cams/{idx:06d}.jpg',
                   cams_all=f'cams/{idx:06d}_all.jpg',
                   pose=[round(float(T_ego_to_map[0, 3]), 3), round(float(T_ego_to_map[1, 3]), 3),
                         round(float(np.arctan2(T_ego_to_map[1, 0], T_ego_to_map[0, 0])), 5)])
        frames.append(rec)
        fjson.write(json.dumps(rec) + '\n')
        fjson.flush()
        if idx % 10 == 0:
            print(f'  frame {idx}: t+{(stamp - t0) / 1e9:6.1f}s  cams {len(raw)}/{len(cams)}  pts {pcd.shape[1]}  '
                  f'{time.time() - tic:.0f}s elapsed', flush=True)

    t0 = None
    wanted = set(cams) | set(infos.values()) | {LIDAR, ODOM, '/tf_static'}
    with open(args.bag, 'rb') as fh:
        for _, ch, _, msg in make_reader(fh, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                topics=sorted(wanted), log_time_order=True):
            t = ch.topic
            if t == '/tf_static':
                for tr in msg.transforms:
                    tf.edges[tr.child_frame_id.lstrip('/')] = (tr.header.frame_id.lstrip('/'),
                                                              tf_matrix(tr.transform.translation, tr.transform.rotation))
                continue
            if t == ODOM:
                p = msg.pose.pose
                odom.append((stamp_ns(msg.header), tf_matrix(p.position, p.orientation)))
                continue
            if t in images:
                images[t].append((stamp_ns(msg.header), msg.header.frame_id.lstrip('/'), bytes(msg.data)))
            elif t.endswith('/camera_info'):
                c = next(c for c, i in infos.items() if i == t)
                K[c] = np.array(msg.p, np.float64).reshape(3, 4)[:, :3]  # rectified images: P's 3x3
            elif t == LIDAR:
                s = stamp_ns(msg.header)
                t0 = t0 or s
                if last_lidar is not None and 0 < s - last_lidar < 2e8:  # one revolution: 50 ms (20 Hz), 100 ms (10 Hz)
                    revolution_ns = s - last_lidar
                last_lidar = s
                if last_sweep is None or s - last_sweep >= sweep_period * 0.9:
                    last_sweep = s
                    key = last_key is None or s - last_key >= period * 0.95
                    last_key = s if key else last_key
                    pending.append((s, msg.header.frame_id.lstrip('/'), read_pointcloud2(msg), key,
                                    args.sweep_ms * 1e6 if args.sweep_ms > 0 else revolution_ns))
                    if key and int((s - t0) / 1e9) % 15 == 0:
                        print(f'  scan t+{(s - t0) / 1e9:.1f}s', flush=True)
            # a keyframe is mapped once the cameras have caught up with it
            newest = max((d[-1][0] for d in images.values() if d), default=0)
            while pending and newest > pending[0][0] + args.max_cam_dt * 1e9:
                process(*pending.popleft())
            if args.max_frames and len(frames) >= args.max_frames:
                break
    for p in pending:
        process(*p)
    fjson.close()

    if tf.edges:
        static_file.write_text(json.dumps([dict(parent=p, child=c, T=T.tolist()) for c, (p, T) in tf.edges.items()]))
    meta = dict(bag=[args.bag], cameras=cams, lidar=LIDAR, base_frame='base_link', ego_frame=ego_frame(),
                pose_source=ODOM, origin=None if origin is None else [float(v) for v in origin],
                front_camera=cams[0] if cams else None, frames=len(frames), hz=args.hz, skipped=dict(skipped),
                map=dict(res_m=0.2, forward_m=[-30, 30], left_m=[-15, 15]))
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'Done: {len(frames)} semantic maps in {out / "maps"} ({time.time() - tic:.0f}s)'
          + (f'  skipped: {dict(skipped)}' if skipped else ''), flush=True)


if __name__ == '__main__':
    main()
