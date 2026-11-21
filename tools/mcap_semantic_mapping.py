#!/usr/bin/env python3
"""Stage 1 of the MCAP pipeline: ROS 2 MCAP (cameras + LiDAR) -> SemVecNet's own semantic BEV maps.

This is tools/nusc_semantic_mapping.py with the nuScenes devkit swapped for an MCAP reader:
  HRNet-Mscale (TensorRT, Mapillary 65 classes) segments every camera image
  -> LiDAR points are projected into each segmented image and take its label
  -> DynamicMap accumulates the labelled points over time in the map frame (Bayes update + confusion matrix)
  -> an ego-centric 60 x 30 m, 0.2 m/px map (300x150 px) is rendered per keyframe.
The maps are what SemVecNet was trained on, so stage 2 (tools/mcap_infer_semvecnet.py) uses them as-is.

Everything the pipeline needs is read from the bag:
  cameras      <ns>/image_rect_compressed (JPEG) + <ns>/camera_info (P or K)
  LiDAR        a sensor_msgs/PointCloud2 topic with x, y, z (+ intensity)
  extrinsics   /tf_static (+ /tf)
  ego pose     a GNSS/INS nav_msgs/Odometry topic if the bag has one (e.g. /novatel/oem7/odom, UTM: odom -> base_link),
               else the /tf chain from a world frame (map/odom/world) to --base-frame, else any Odometry topic
The map is centred on --ego-frame (default: rear_axle if the TF tree has it), like nuScenes' rear-axle ego frame.
"""
import argparse
import bisect
import json
import os
import queue
import threading
import os.path as osp
import re
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, REPO)

from mcap.reader import make_reader  # noqa: E402
from mcap_ros2.decoder import DecoderFactory  # noqa: E402

IMAGE_SUFFIX = '/image_rect_compressed'
WORLD_FRAMES = ('map', 'odom', 'world', 'earth', 'utm')
NETWORK_HW = (1440 // 2, 1920 // 2)  # HRNet engine input (H, W), as in nusc_semantic_mapping.py

# --------------------------------------------------------------------------- bag helpers


def bag_files(path):
    """A .mcap file, or a rosbag2 directory of split files (<name>_0.mcap, _1, ...) in order."""
    p = Path(path)
    if p.is_file():
        return [p]
    files = list(p.glob('*.mcap'))
    if not files:
        sys.exit(f'No .mcap files in {p}')
    split = lambda f: int(m.group(1)) if (m := re.search(r'_(\d+)\.mcap$', f.name)) else 0
    return sorted(files, key=lambda f: (split(f), f.name))


def list_topics(files):
    topics = {}
    for f in files:
        with open(f, 'rb') as fh:
            summary = make_reader(fh).get_summary()
        counts = summary.statistics.channel_message_counts if summary.statistics else {}
        for cid, ch in summary.channels.items():
            name = summary.schemas[ch.schema_id].name if ch.schema_id in summary.schemas else ''
            t = topics.setdefault(ch.topic, dict(type=name, count=0))
            t['count'] += counts.get(cid, 0)
    return topics


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def tf_matrix(t, q):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


_PC2_DTYPES = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


def read_pointcloud2(msg):
    """PointCloud2 -> (4, N) float array of x, y, z, intensity (intensity 0 if absent)."""
    fields = {f.name: f for f in msg.fields}
    names = [n for n in ('x', 'y', 'z', 'intensity') if n in fields]
    dtype = np.dtype(dict(names=names, formats=[('>' if msg.is_bigendian else '<') + _PC2_DTYPES[fields[n].datatype] for n in names],
                          offsets=[fields[n].offset for n in names], itemsize=msg.point_step))
    n_pts = msg.width * msg.height
    rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.row_step)[:, :msg.width * msg.point_step]
    arr = np.frombuffer(rows.tobytes(), dtype=dtype, count=n_pts)
    pcd = np.zeros((4, n_pts), np.float64)
    for i, n in enumerate(('x', 'y', 'z', 'intensity')):
        if n in arr.dtype.names:
            pcd[i] = arr[n]
    return pcd[:, np.isfinite(pcd[:3]).all(0) & (np.abs(pcd[:3]).sum(0) > 0.1)]


# --------------------------------------------------------------------------- transforms


class TfBuffer:
    """Frame graph from /tf_static, /tf and (optionally) an odometry topic.

    Every edge keeps a short time history; edges with a single sample (static TF, or /tf edges that are
    published once, like the sensor mounts in the nuScenes bags) are used at any time.
    """

    def __init__(self, horizon_ns=30_000_000_000):
        self.edges = {}  # child -> (parent, stamps[], mats[])
        self.static = set()
        self.horizon = horizon_ns

    def add(self, parent, child, stamp, T, static=False):
        parent, child = parent.lstrip('/'), child.lstrip('/')
        if static:
            self.static.add(child)
            self.edges[child] = (parent, [0], [T])
            return
        if child in self.static:
            return
        p, st, ms = self.edges.setdefault(child, (parent, [], []))
        if st and stamp <= st[-1]:
            if stamp == st[-1]:
                return
            i = bisect.bisect(st, stamp)
            st.insert(i, stamp)
            ms.insert(i, T)
        else:
            st.append(stamp)
            ms.append(T)
        while len(st) > 2 and st[-1] - st[0] > self.horizon:
            st.pop(0)
            ms.pop(0)

    def latest(self, child):
        return self.edges[child][1][-1] if child in self.edges else None

    def gap_ns(self, child, t):
        """Time between the samples of a moving edge around t (0 for static edges, inf if t is not bracketed)."""
        if child not in self.edges:
            return float('inf')
        _, st, _ = self.edges[child]
        if len(st) == 1 and st[0] == 0:
            return 0
        i = bisect.bisect_left(st, t)
        if i < len(st) and st[i] == t:
            return 0
        if i == 0 or i == len(st):
            return float('inf')
        return st[i] - st[i - 1]

    def _edge(self, child, t):
        parent, st, ms = self.edges[child]
        if len(st) == 1:
            return parent, ms[0]
        i = bisect.bisect_left(st, t)
        if i == 0:
            return parent, ms[0]
        if i == len(st):
            return parent, ms[-1]
        a, b = st[i - 1], st[i]
        w = (t - a) / (b - a)
        T = np.eye(4)
        T[:3, 3] = (1 - w) * ms[i - 1][:3, 3] + w * ms[i][:3, 3]
        rots = Rotation.from_matrix(np.stack([ms[i - 1][:3, :3], ms[i][:3, :3]]))
        T[:3, :3] = Slerp([0, 1], rots)([w]).as_matrix()[0]
        return parent, T

    def to_root(self, frame, t):
        """(root frame, T_root_frame) at time t."""
        frame = frame.lstrip('/')
        T = np.eye(4)
        seen = set()
        while frame in self.edges and frame not in seen:
            seen.add(frame)
            parent, E = self._edge(frame, t)
            T = E @ T
            frame = parent
        return frame, T

    def lookup(self, target, source, t):
        """T_target_source (maps points in `source` into `target`)."""
        r1, T_r_t = self.to_root(target, t)
        r2, T_r_s = self.to_root(source, t)
        if r1 != r2:
            raise KeyError(f'no TF path between {target} and {source} (roots {r1}, {r2})')
        return np.linalg.inv(T_r_t) @ T_r_s

    def has_path(self, a, b):
        return self.to_root(a, 0)[0] == self.to_root(b, 0)[0]


# --------------------------------------------------------------------------- segmentation


class Segmenter:
    """HRNet TensorRT segmentation, same pre/post-processing as get_semantic_image() in nusc_semantic_mapping.py."""

    def __init__(self, cfg):
        import semantic_mapping.src.utils.mapillary_visualization as mapillary_visl
        from semantic_mapping.src.hrnet.hrnet_semantic_segmentation_tensorrt import (
            HRNetSemanticSegmentationTensorRT, get_custom_hrnet_args)
        self.visl = mapillary_visl
        self.color_ref = mapillary_visl.get_labels(cfg.VISION_SEM_SEG.DATASET_CONFIG)
        self.net = HRNetSemanticSegmentationTensorRT(get_custom_hrnet_args(cfg))

    # split in three so CPU work (pre / post, any thread) overlaps the GPU work (infer, the CUDA context's thread)
    def pre(self, image_rgb):
        h, w = NETWORK_HW
        return self.net.preprocess(cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA))

    def infer(self, x):
        h, w = NETWORK_HW
        return np.reshape(self.net.segmentation(x), (h, w)).astype(np.uint8)

    def post(self, labels, shape):
        colored = np.squeeze(self.visl.apply_color_map(labels, self.color_ref)).astype(np.uint8)
        return cv2.resize(colored, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    def __call__(self, image_rgb):
        return self.post(self.infer(self.pre(image_rgb)), image_rgb.shape)

    def decode_pre(self, jpeg):
        img = cv2.imdecode(np.frombuffer(bytes(jpeg), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, self.pre(img)


# --------------------------------------------------------------------------- camera labels on the ground


def fit_ground_plane(pts_ground):
    """Road plane z = a x + b y + c (ground frame) from LiDAR points 4-20 m away; falls back to z = 0."""
    x, y, z = pts_ground
    r = np.hypot(x, y)
    m = (r > 4) & (r < 20) & (np.abs(z) < 0.6)
    if m.sum() < 200:
        return np.array([0.0, 0.0, 0.0])
    A = np.stack([x[m], y[m], np.ones(m.sum())], 1)
    zz = z[m]
    coef = np.array([0.0, 0.0, 0.0])
    keep = np.ones(len(zz), bool)
    for _ in range(3):  # least squares, then drop points off the plane (curbs, cars, grass)
        coef, *_ = np.linalg.lstsq(A[keep], zz[keep], rcond=None)
        keep = np.abs(A @ coef - zz) < 0.12
        if keep.sum() < 100:
            return np.array([0.0, 0.0, 0.0])
    if abs(coef[0]) > 0.1 or abs(coef[1]) > 0.1 or abs(coef[2]) > 0.4:  # > ~6 deg or 40 cm: not trusted
        return np.array([0.0, 0.0, 0.0])
    return coef


class GroundProjector:
    """Where each (sub-sampled) pixel of one camera meets the road plane, in the ground frame.

    The rays only depend on the intrinsics and the (static) mount, so they are computed once; per frame only the
    intersection with that frame's road plane is solved.
    """

    def __init__(self, K, wh, T_ground_cam, stride, r_min, r_max):
        w, h = wh
        us, vs = np.meshgrid(np.arange(stride // 2, w, stride), np.arange(stride // 2, h, stride))
        rays = np.linalg.inv(K) @ np.stack([us.ravel(), vs.ravel(), np.ones(us.size)])
        d = T_ground_cam[:3, :3] @ rays
        down = d[2] < -1e-3  # only pixels looking down can hit the road
        self.u, self.v, self.d = us.ravel()[down], vs.ravel()[down], d[:, down]
        self.o = T_ground_cam[:3, 3]
        self.r_min, self.r_max = r_min, r_max

    def __call__(self, sem_img, plane):
        a, b, c = plane  # ray o + s d meets z = a x + b y + c
        denom = self.d[2] - a * self.d[0] - b * self.d[1]
        s = (a * self.o[0] + b * self.o[1] + c - self.o[2]) / np.where(np.abs(denom) < 1e-6, -1e-6, denom)
        p = self.o[:, None] + self.d * s
        r = np.hypot(p[0], p[1])
        ok = (s > 0) & (r >= self.r_min) & (r <= self.r_max)
        labels = sem_img[self.v[ok], self.u[ok]].T.copy()
        # intensity 5: neutral for DynamicMap's LiDAR-intensity lane boost (< 2 or > 14)
        pts = np.vstack([p[:, ok], np.full((1, ok.sum()), 5.0)])
        return pts, labels


# --------------------------------------------------------------------------- main loop


def pick_lidar(topics, wanted):
    if wanted:
        return wanted
    clouds = {t: v['count'] for t, v in topics.items() if v['type'].endswith('PointCloud2') and v['count'] > 0
              and not re.search(r'radar|filtered|ground|nonground|cluster|deskew|submap|visualization', t, re.I)}
    if not clouds:
        sys.exit('No PointCloud2 topic with messages in the bag; pass --lidar-topic')
    # all LiDARs fused (roof + mirror units: twice the points and the near field the roof unit cannot see), else
    # a roof-centre LiDAR like nuScenes' LIDAR_TOP; corner / mirror LiDARs only as a fallback
    for pattern in (r'all/points_merged|points_merged', r'LIDAR_TOP|lidar_cc|lidar_top|/top', r'velodyne_points|points_raw|lidar'):
        pref = [t for t in clouds if re.search(pattern, t, re.I)]
        if pref:
            return max(pref, key=lambda t: clouds[t])
    return max(clouds, key=lambda t: clouds[t])


def pick_cameras(topics, wanted):
    if wanted:
        return [c if c.endswith(IMAGE_SUFFIX) else c.rstrip('/') + IMAGE_SUFFIX for c in wanted.split(',')]
    cams = sorted(t for t, v in topics.items() if t.endswith(IMAGE_SUFFIX) and v['type'].endswith('CompressedImage')
                  and v['count'] > 0 and t[:-len(IMAGE_SUFFIX)] + '/camera_info' in topics)
    if not cams:
        sys.exit(f'No <camera>{IMAGE_SUFFIX} topics with a camera_info in the bag; pass --cameras')
    return cams


def pick_odom(topics, wanted):
    if wanted:
        return wanted
    odoms = {t: v['count'] for t, v in topics.items() if v['type'].endswith('Odometry') and v['count'] > 0
             and not t.endswith('_origin')}
    gnss = [t for t in odoms if is_gnss(t)]  # georeferenced (UTM) pose, needed to compare with an HD map
    return max(gnss or odoms, key=lambda t: odoms[t]) if odoms else None


def is_gnss(topic):
    return bool(topic and re.search(r'novatel|gnss|gps|ins\b|oem7', topic, re.I))


def make_cfg(args):
    from semantic_mapping.src.node_config.nuscenes_cfg import get_cfg_defaults
    cfg = get_cfg_defaults()
    cfg.MAPPING.CONFUSION_MTX.LOAD_PATH = args.confusion_matrix
    cfg.VISION_SEM_SEG.DATASET_CONFIG = args.seg_config
    cfg.VISION_SEM_SEG.ENGINE_FILE = args.engine
    cfg.VISION_SEM_SEG.DUMMY_IMAGE_PATH = args.dummy_image
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bag', required=True, help='.mcap file or rosbag2 directory')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cameras', default='', help='comma list of camera namespaces/topics (default: every '
                    f'<ns>{IMAGE_SUFFIX} with a camera_info)')
    ap.add_argument('--lidar-topic', default='')
    ap.add_argument('--odom-topic', default='', help='ego pose topic, used when /tf has no world frame above '
                    '--base-frame (default: the Odometry topic with the most messages)')
    ap.add_argument('--pose-source', choices=('auto', 'tf', 'odom'), default='auto')
    ap.add_argument('--base-frame', default='base_link', help='vehicle frame the pose refers to')
    ap.add_argument('--ego-frame', default='auto', help='map centre (default: rear_axle if in TF, else --base-frame)')
    ap.add_argument('--hz', type=float, default=2.0, help='keyframe rate (nuScenes keyframes are 2 Hz)')
    ap.add_argument('--map-hz', type=float, default=10.0,
                    help='rate at which LiDAR sweeps are labelled and accumulated into the map (>= --hz); '
                         'keyframes are still written at --hz. 2 = the original nuScenes recipe (keyframes only)')
    ap.add_argument('--start', type=float, default=0.0, help='seconds to skip from the bag start')
    ap.add_argument('--duration', type=float, default=0.0, help='seconds to process (0 = all)')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--max-cam-dt', type=float, default=0.1, help='max |camera - lidar| stamp gap, seconds')
    ap.add_argument('--max-pose-gap', type=float, default=0.2,
                    help='skip keyframes whose pose would be interpolated across a gap longer than this, seconds')
    ap.add_argument('--camera-ground', type=int, default=1,
                    help='1: also project every camera pixel\'s label onto the road plane (dense paint / crosswalks '
                         'from the cameras, not only where LiDAR points land); 0: original LiDAR-only labelling')
    ap.add_argument('--camera-range', type=float, default=25.0, help='max distance of camera ground labels, m')
    ap.add_argument('--camera-classes', default='all', help='classes taken from the camera ground labels: all, or a '
                    'comma list of road,crosswalk,lane,sidewalk (e.g. lane,crosswalk = painted markings only)')
    ap.add_argument('--camera-stride', type=int, default=4, help='pixel sub-sampling of camera ground labels')
    ap.add_argument('--ground-frame', default='auto', help='frame whose z = 0 is the road (default: base_footprint '
                    'if in TF, else the ego frame)')
    ap.add_argument('--engine', required=True)
    ap.add_argument('--seg-config', default=osp.join(REPO, 'semantic_mapping/config/config_65.json'))
    ap.add_argument('--confusion-matrix', default=osp.join(REPO, 'semantic_mapping/src/hrnet/assets/seg_weights/hrnet_cfn_1999.npy'))
    ap.add_argument('--dummy-image', default=osp.join(REPO, 'semantic_mapping/src/hrnet/assets/seg_weights/1682354962.572154522.jpg'))
    args = ap.parse_args()
    for k in ('bag', 'out', 'engine', 'seg_config', 'confusion_matrix', 'dummy_image'):
        setattr(args, k, osp.abspath(getattr(args, k)))
    os.chdir(REPO)  # semantic_mapping/src/utils/mapillary_visualization.py opens a path relative to the repo root

    files = bag_files(args.bag)
    topics = list_topics(files)
    cams = pick_cameras(topics, args.cameras)
    lidar_topic = pick_lidar(topics, args.lidar_topic)
    info_topics = {c[:-len(IMAGE_SUFFIX)] + '/camera_info': c for c in cams}
    odom_topic = pick_odom(topics, args.odom_topic)
    print(f'Bag: {", ".join(f.name for f in files)}')
    print(f'Cameras ({len(cams)}): {", ".join(cams)}')
    print(f'LiDAR: {lidar_topic}   odometry fallback: {odom_topic}', flush=True)

    out = Path(args.out)
    for d in ('maps', 'cams', 'inputs'):
        (out / d).mkdir(parents=True, exist_ok=True)

    cfg = make_cfg(args)
    from semantic_mapping.src.dynamic_map import DynamicMap
    seg = Segmenter(cfg)
    dm = DynamicMap(cfg)

    tf = TfBuffer()     # sensor mounts (+ world -> vehicle if the bag publishes it on /tf)
    odom = TfBuffer()   # odometry topic poses, kept apart so they never clash with the static tree
    odom_child = None
    infos = {}  # camera topic -> (K 3x3, (w, h))
    images = defaultdict(lambda: deque(maxlen=60))  # camera topic -> (stamp, frame_id, jpeg bytes)
    last_cam_stamp = defaultdict(int)
    pending = deque()  # (stamp, frame_id, pcd) keyframes waiting for cameras and pose
    state = dict(pose_source=None, ego_frame=None, ground_frame=None)
    projectors = {}
    pool = ThreadPoolExecutor(max_workers=6)   # JPEG decode, HRNet pre/post-processing
    mapper = ThreadPoolExecutor(max_workers=1)  # DynamicMap updates + outputs, in order
    map_jobs = deque()
    cam_colors = None if args.camera_classes == 'all' else np.array(
        [cfg.LABEL_COLORS[cfg.LABELS_NAMES.index(n.strip())] for n in args.camera_classes.split(',')], np.uint8)

    def ground_frame():
        if state['ground_frame'] is None:
            g = args.ground_frame
            if g == 'auto':
                g = 'base_footprint' if 'base_footprint' in tf.edges or any(
                    p == 'base_footprint' for p, _, _ in tf.edges.values()) else ego_frame()
            state['ground_frame'] = g
            print(f'  camera labels projected onto the road plane of {g} (z = 0, refit to the LiDAR every sweep)', flush=True)
        return state['ground_frame']
    period = int(1e9 / args.hz)
    map_period = int(1e9 / max(args.map_hz, args.hz))
    last_map = -map_period
    t0 = None
    last_key = -period
    origin = None
    front_cam = None
    frames = []
    n_skipped = defaultdict(int)
    tic = time.time()

    def pose_source():
        if state['pose_source'] is None and (tf.edges or odom.edges):
            root = tf.to_root(args.base_frame, 0)[0]
            if args.pose_source != 'auto':
                state['pose_source'] = args.pose_source
            elif is_gnss(odom_topic):  # prefer the georeferenced GNSS/INS pose over SLAM / local /tf frames
                state['pose_source'] = 'odom'
            elif root in WORLD_FRAMES and root != args.base_frame:
                state['pose_source'] = 'tf'
            elif odom_topic:
                state['pose_source'] = 'odom'
            else:
                sys.exit(f'No ego pose: /tf has no world frame above {args.base_frame} (root: {root}) and the bag '
                         'has no Odometry topic. Pass --odom-topic or --pose-source.')
            print(f'  ego pose from {"/tf (" + root + " -> " + args.base_frame + ")" if state["pose_source"] == "tf" else odom_topic}',
                  flush=True)
        return state['pose_source']

    def pose_ready(stamp):
        src = pose_source()
        if src is None:
            return False
        latest = tf.latest(args.base_frame) if src == 'tf' else (odom.latest(odom_child) if odom_child else None)
        return latest is not None and latest >= stamp

    def ego_frame():
        if state['ego_frame'] is None:
            e = args.ego_frame
            if e == 'auto':
                e = 'rear_axle' if 'rear_axle' in tf.edges and tf.has_path('rear_axle', args.base_frame) else args.base_frame
            state['ego_frame'] = e
            print(f'  map centred on {e}', flush=True)
        return state['ego_frame']

    def world_T(frame, stamp):
        """T_world_frame at `stamp` for any frame in the vehicle's static tree."""
        if pose_source() == 'tf':
            return tf.to_root(frame, stamp)[1]
        return odom.to_root(odom_child, stamp)[1] @ tf.lookup(odom_child, frame, stamp)

    def pose_gap_ns(stamp):
        if pose_source() == 'tf':
            return tf.gap_ns(args.base_frame, stamp)
        return odom.gap_ns(odom_child, stamp) if odom_child else float('inf')

    def snapshot(stamp):
        """Nearest image of every camera to a LiDAR stamp, taken as soon as the cameras have caught up."""
        snap = {}
        for c in cams:
            if not images[c]:
                continue
            s, frame_id, data = min(images[c], key=lambda m: abs(m[0] - stamp))
            if abs(s - stamp) <= args.max_cam_dt * 1e9:
                snap[c] = (s, frame_id, data)
        return snap

    def process(stamp, lidar_frame, pcd, snap, key=True):
        """Main thread: poses + TF (the TF buffers are filled by this thread) and HRNet on the GPU (the CUDA
        context lives here). JPEG decode / pre- and post-processing run in `pool`, and everything after
        segmentation runs in the ordered `mapper` thread, so the GPU segments sweep N+1 while sweep N is mapped."""
        nonlocal origin, front_cam
        # only map with a measured pose: interpolating across a pose dropout (NovAtel has ~10 s gaps in some
        # bags) would put the points, and the ground truth patch, metres away from where they belong
        if pose_gap_ns(stamp) > args.max_pose_gap * 1e9:
            n_skipped['pose dropout (keyframes)' if key else 'pose dropout (map-only sweeps)'] += 1
            return
        T_map_ego = world_T(ego_frame(), stamp)
        T_map_lidar = world_T(lidar_frame, stamp)
        if origin is None:  # keep map coordinates small: start the map frame at the first pose
            origin = T_map_ego[:3, 3].copy()
        T_map_ego[:3, 3] -= origin
        T_map_lidar[:3, 3] -= origin

        jobs = {}
        for c in cams:
            if c not in infos or c not in snap:
                n_skipped['camera image missing / too far in time'] += 1
                continue
            s, frame_id, data = snap[c]
            try:
                T_cam_lidar = tf.lookup(frame_id, lidar_frame, stamp)
            except KeyError:
                n_skipped[f'no TF {frame_id}->{lidar_frame}'] += 1
                continue
            jobs[c] = (T_cam_lidar, frame_id, pool.submit(seg.decode_pre, data))
        if not jobs:
            return
        if front_cam is None:  # camera whose optical axis points most along the ego's +x
            fwd = {cc: tf.lookup(ego_frame(), images[cc][0][1], stamp)[0, 2]
                   for cc in cams if images[cc] and tf.has_path(ego_frame(), images[cc][0][1])}
            front_cam = max(fwd, key=fwd.get) if fwd else next(iter(jobs))
        ctx = dict(stamp=stamp, key=key, pcd=pcd, T_map_ego=T_map_ego, T_map_lidar=T_map_lidar, front_cam=front_cam,
                   T_l2c={c: j[0] for c, j in jobs.items()}, cams={})
        if args.camera_ground:
            gf = ground_frame()
            ctx['T_g_lidar'] = tf.lookup(gf, lidar_frame, stamp)
            ctx['T_map_ground'] = world_T(gf, stamp)
            ctx['T_map_ground'][:3, 3] -= origin
            ctx['T_g_cam'] = {c: tf.lookup(gf, j[1], stamp) for c, j in jobs.items() if c not in projectors}
        for c, (_, _, fut) in jobs.items():
            img, x = fut.result()
            if img is None:
                continue
            labels = seg.infer(x)  # GPU
            ctx['cams'][c] = (img, pool.submit(seg.post, labels, img.shape))
        map_jobs.append(mapper.submit(map_step, ctx))
        while len(map_jobs) > 2:  # bound memory, and surface errors from the mapping thread
            map_jobs.popleft().result()

    def map_step(ctx):
        """Mapping thread (strictly in order): LiDAR / camera labels -> DynamicMap, then the keyframe outputs."""
        stamp, key, pcd = ctx['stamp'], ctx['key'], ctx['pcd']
        T_map_ego, T_map_lidar = ctx['T_map_ego'], ctx['T_map_lidar']
        sem, intr, T_l2c, used, raw_imgs = {}, {}, {}, [], {}
        front = None
        for c, (img, fut) in ctx['cams'].items():
            K, (w, h) = infos[c]
            if (w, h) != (img.shape[1], img.shape[0]) and w > 0:  # image was resized after calibration
                K = np.diag([img.shape[1] / w, img.shape[0] / h, 1.0]) @ K
            sem[c], intr[c], T_l2c[c] = fut.result(), K, ctx['T_l2c'][c]
            raw_imgs[c] = img
            used.append(c)
            if c == ctx['front_cam']:
                front = (0.5 * img + 0.5 * sem[c]).astype(np.uint8)
        if not used:
            return
        if not args.camera_ground:
            _, map_rendered, _, _ = dm.mapping_nuscenes(pcd, sem, used, intr, T_l2c, T_map_lidar, T_map_ego)
        else:
            # same LiDAR labelling + Bayes update as DynamicMap.mapping_nuscenes ...
            dm.check_map_origin(T_map_lidar)
            P, L = [], []
            for c in used:
                p_, l_ = dm.project_pcd(pcd, sem[c], intr[c], T_l2c[c])
                P.append(p_)
                L.append(dm.merge_color(l_))
            dm.map = dm.update_map(dm.map, np.concatenate(P, 1), np.concatenate(L, 1), T_map_lidar)
            # ... plus the camera labels where each pixel meets the road plane (fitted to this sweep's LiDAR)
            T_g_lidar = ctx['T_g_lidar']
            plane = fit_ground_plane(T_g_lidar[:3, :3] @ pcd[:3] + T_g_lidar[:3, 3:4])
            P, L = [], []
            for c in used:
                if c not in projectors:
                    projectors[c] = GroundProjector(intr[c], (sem[c].shape[1], sem[c].shape[0]), ctx['T_g_cam'][c],
                                                    args.camera_stride, 2.0, args.camera_range)
                p_, l_ = projectors[c](sem[c], plane)
                l_ = dm.merge_color(l_)
                if cam_colors is not None:
                    sel = (l_.T[:, None, :] == cam_colors[None]).all(2).any(1)
                    p_, l_ = p_[:, sel], l_[:, sel]
                P.append(p_)
                L.append(l_)
            dm.map = dm.update_map(dm.map, np.concatenate(P, 1), np.concatenate(L, 1), ctx['T_map_ground'])
            map_rendered = dm.render_local_map(dm.map, T_map_ego, dm.render_ego_centric, filter=False)
        if not key:  # sweep only densifies the map
            return

        idx = len(frames)
        # same convention as save_semantic_image(..., RGB2BGR=True) in nusc_semantic_mapping.py
        cv2.imwrite(str(out / 'maps' / f'{idx:06d}.png'), cv2.cvtColor(map_rendered.astype(np.uint8), cv2.COLOR_BGR2RGB))
        rec = dict(idx=idx, stamp_ns=stamp, map=f'maps/{idx:06d}.png', fps=args.hz, cams=len(used),
                   pose=[round(float(T_map_ego[0, 3]), 3), round(float(T_map_ego[1, 3]), 3),
                         round(float(np.arctan2(T_map_ego[1, 0], T_map_ego[0, 0])), 5)])
        # the raw inputs of this keyframe, for tools/mcap_compare_viz.py: every camera image, and the LiDAR sweep
        # in the ego frame (x fwd, y left, z up, intensity) cropped to the map area
        inp = dict(cams={})
        for c, im in raw_imgs.items():
            name = c[:-len(IMAGE_SUFFIX)].strip('/').replace('/', '_')
            small = cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), (480, int(480 * im.shape[0] / im.shape[1])),
                               interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / 'inputs' / f'{idx:06d}_{name}.jpg'), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
            inp['cams'][c] = f'inputs/{idx:06d}_{name}.jpg'
        T_ego_lidar = np.linalg.inv(T_map_ego) @ T_map_lidar
        pts = (T_ego_lidar[:3, :3] @ pcd[:3] + T_ego_lidar[:3, 3:4])
        keep = (np.abs(pts[0]) < 32) & (np.abs(pts[1]) < 17)
        np.save(out / 'inputs' / f'{idx:06d}_lidar.npy', np.vstack([pts[:, keep], pcd[3:4, keep]]).T.astype(np.float16))
        inp['lidar'] = f'inputs/{idx:06d}_lidar.npy'
        rec['inputs'] = inp
        if front is not None:
            f = cv2.cvtColor(front, cv2.COLOR_RGB2BGR)
            f = cv2.resize(f, (640, int(640 * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / 'cams' / f'{idx:06d}.jpg'), f, [cv2.IMWRITE_JPEG_QUALITY, 85])
            rec['front'] = f'cams/{idx:06d}.jpg'
        frames.append(rec)
        with open(out / 'frames.jsonl', 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        if idx % 10 == 0:
            print(f'  frame {idx}: t+{(stamp - t0) / 1e9:6.1f}s  cams {len(used)}/{len(cams)}  '
                  f'pts {pcd.shape[1]}  {time.time() - tic:.0f}s elapsed', flush=True)

    def drain(force=False):
        # 1) give every waiting keyframe its camera images as soon as the cameras have caught up with it,
        #    so a keyframe that then waits for its pose (e.g. through a GNSS dropout) keeps its images
        for item in pending:
            if item[3] is not None:
                continue
            stamp = item[0]
            # wait for every camera that is still publishing (one that went silent >2 s ago is not waited on)
            cams_ready = all(last_cam_stamp[c] >= stamp + args.max_cam_dt * 1e9
                             for c in cams if last_cam_stamp[c] > stamp - 2_000_000_000)
            if not (force or (cams_ready and any(last_cam_stamp[c] > 0 for c in cams))):
                break
            item[3] = snapshot(stamp)
        # 2) map keyframes in order once their pose is known
        while pending and pending[0][3] is not None:
            stamp, lf, pcd, snap, key = pending[0]
            # wait for the pose, and for /tf_static to link the LiDAR to the vehicle: a split file from the middle of
            # a recording starts without /tf_static (it is only re-published about once a second)
            if not force and not (pose_ready(stamp) and tf.has_path(args.base_frame, lf)):
                break
            pending.popleft()
            if pose_source() is None or (pose_source() == 'odom' and odom_child is None):
                n_skipped['no ego pose'] += 1
                continue
            if not tf.has_path(args.base_frame, lf):
                n_skipped[f'no TF {args.base_frame}->{lf}'] += 1
                continue
            process(stamp, lf, pcd, snap, key)
            if args.max_frames and len(frames) >= args.max_frames:
                return True
        return False

    def camera_azimuths():
        """Direction of each camera's optical axis in the ego frame (0 = forward, +90 = left), for display order."""
        az = {}
        for c in cams:
            if images[c] and state['ego_frame'] and tf.has_path(state['ego_frame'], images[c][0][1]):
                R = tf.lookup(state['ego_frame'], images[c][0][1], 0)[:3, :3]
                az[c] = round(float(np.degrees(np.arctan2(R[1, 2], R[0, 2]))), 1)
        return az

    (out / 'frames.jsonl').write_text('')
    wanted = set(cams) | set(info_topics) | {lidar_topic, '/tf', '/tf_static'} | ({odom_topic} if odom_topic else set())
    # the bag is read and CDR-decoded in its own thread, so it runs while this thread waits on the GPU
    msg_q = queue.Queue(maxsize=256)
    stop = threading.Event()

    def produce():
        try:
            for f in files:
                with open(f, 'rb') as fh:
                    reader = make_reader(fh, decoder_factories=[DecoderFactory()])
                    for _, ch, _, ros in reader.iter_decoded_messages(topics=sorted(wanted), log_time_order=True):
                        if stop.is_set():
                            return
                        msg_q.put((ch.topic, ros))
        except Exception as e:  # surfaced by the consumer
            msg_q.put(('__error__', e))
        finally:
            msg_q.put(None)

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    while True:
        item = msg_q.get()
        if item is None:
            break
        if item[0] == '__error__':
            raise item[1]
        topic, ros = item
        if topic in ('/tf', '/tf_static'):
            for tr in ros.transforms:
                tf.add(tr.header.frame_id, tr.child_frame_id, stamp_ns(tr.header),
                       tf_matrix(tr.transform.translation, tr.transform.rotation), static=topic == '/tf_static')
            continue
        if topic == odom_topic:
            odom_child = (ros.child_frame_id or args.base_frame).lstrip('/')
            p = ros.pose.pose
            odom.add(ros.header.frame_id or 'odom', odom_child, stamp_ns(ros.header),
                     tf_matrix(p.position, p.orientation))
            continue
        if topic in info_topics:
            c = info_topics[topic]
            if c not in infos:
                P = np.array(ros.p, np.float64).reshape(3, 4)
                K = P[:, :3] if np.abs(P).sum() > 0 else np.array(ros.k, np.float64).reshape(3, 3)
                infos[c] = (K, (int(ros.width), int(ros.height)))
            continue
        stamp = stamp_ns(ros.header)
        if t0 is None:
            t0 = stamp
        if topic == lidar_topic:
            rel = (stamp - t0) / 1e9
            if rel < args.start or stamp - last_map < 0.9 * map_period:
                continue
            if args.duration and rel > args.start + args.duration:
                break
            last_map = stamp
            key = stamp - last_key >= 0.9 * period
            if key:
                last_key = stamp
            pending.append([stamp, ros.header.frame_id, read_pointcloud2(ros), None, key])
        else:  # camera image
            images[topic].append((stamp, ros.header.frame_id, ros.data))
            last_cam_stamp[topic] = max(last_cam_stamp[topic], stamp)
        if drain():
            break
    stop.set()
    while producer.is_alive():  # unblock the producer if it is waiting on a full queue
        try:
            msg_q.get(timeout=0.1)
        except queue.Empty:
            pass
    drain(force=True)
    for j in map_jobs:
        j.result()
    pool.shutdown()
    mapper.shutdown()

    meta = dict(bag=[str(f) for f in files], cameras=cams, lidar=lidar_topic, base_frame=args.base_frame,
                ego_frame=state['ego_frame'], pose_source=odom_topic if state['pose_source'] == 'odom' else '/tf',
                # frames.jsonl poses are relative to this point of the pose source's world frame (UTM for NovAtel odom)
                origin=None if origin is None else [float(v) for v in origin],
                front_camera=front_cam, frames=len(frames),
                camera_azimuth_deg=camera_azimuths(),
                hz=args.hz, skipped=dict(n_skipped), map=dict(res_m=0.2, forward_m=[-30, 30], left_m=[-15, 15]))
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'Done: {len(frames)} semantic maps in {out / "maps"} ({time.time() - tic:.0f}s)'
          + (f'  skipped: {dict(n_skipped)}' if n_skipped else ''))
    if not frames:
        sys.exit(1)


if __name__ == '__main__':
    main()
