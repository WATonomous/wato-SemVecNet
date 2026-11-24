#!/usr/bin/env python3
"""Check (and correct) the camera mounts in a bag's /tf_static against its LiDAR.

LiDAR depth edges (where the range jumps between neighbouring points of a ring: poles, car and wall outlines) are
projected into every camera - motion compensated like tools/mcap_semantic_mapping.py - and compared with the image's
own edges. A correct mount puts the LiDAR edges on image edges. For each camera a small rotation of the mount
(pitch / yaw / roll about the camera's x / y / z axes) is searched that lines them up best.

Writes in --out:
  calibration.json    camera frame -> 4x4 correction (mcap_semantic_mapping.py --camera-calibration)
  report.txt          per camera: edge alignment with the /tf_static mount and with the correction, and the angles
  overlay_<cam>.jpg   LiDAR points (by depth) on one image: /tf_static mount | corrected
"""
import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcap_semantic_mapping import LIDAR, ODOM, PANO_GRID, StaticTf, stamp_ns, tf_matrix  # noqa: E402

KINDS = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
WIDTH = 640  # images are compared at this width


def lidar_edges(msg):
    """Foreground points at range discontinuities along each ring: (3, N) in the LiDAR frame."""
    f = {x.name: x for x in msg.fields}
    names = [n for n in ('x', 'y', 'z', 'ring') if n in f]
    dt = np.dtype(dict(names=names, formats=['<' + KINDS[f[n].datatype] for n in names],
                       offsets=[f[n].offset for n in names], itemsize=msg.point_step))
    a = np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)
    p = np.stack([a['x'], a['y'], a['z']]).astype(np.float64)
    ok = np.isfinite(p).all(0) & (np.linalg.norm(p, axis=0) > 1.0)
    p = p[:, ok]
    r = np.linalg.norm(p, axis=0)
    ring = a['ring'][ok] if 'ring' in names else np.round(np.degrees(np.arctan2(p[2], np.hypot(p[0], p[1])))).astype(int)
    az = np.arctan2(p[1], p[0])
    out = []
    for k in np.unique(ring):
        i = np.where(ring == k)[0]
        i = i[np.argsort(az[i])]
        if len(i) < 3:
            continue
        rr = r[i]
        jump_prev = np.r_[0, rr[:-1] - rr[1:]]  # > 0: previous point farther -> this one is a foreground edge
        jump_next = np.r_[rr[1:] - rr[:-1], 0]
        edge = ((jump_prev > 0.5) | (jump_next > 0.5)) & (rr < 40)
        out.append(p[:, i[edge]])
    return np.concatenate(out, 1) if out else np.zeros((3, 0))


def rot(rpy_deg):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', rpy_deg, degrees=True).as_matrix()  # camera axes: x right, y down, z fwd
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', type=int, default=40, help='sweeps used (1 per second while the car moves)')
    ap.add_argument('--start', type=float, default=0.0, help='seconds to skip from the bag start')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.bag, 'rb') as fh:
        topics = {ch.topic for ch in make_reader(fh).get_summary().channels.values()}
    cams = [f'/camera_pano_{c}/image_rect_compressed' for row in PANO_GRID for c in row
            if c and f'/camera_pano_{c}/image_rect_compressed' in topics]
    infos = {c.rsplit('/', 1)[0] + '/camera_info': c for c in cams}
    tf, K, size = StaticTf(), {}, {}
    images = {c: deque(maxlen=30) for c in cams}
    odom = deque(maxlen=400)
    samples = {c: [] for c in cams}  # (edge points in base frame at image time, edge distance image, image)
    pending, last, t0 = deque(), None, None

    def pose_at(t):
        after = next((i for i, (s, _) in enumerate(odom) if s >= t), None)
        if not after:
            return None
        (s0, T0), (s1, T1) = odom[after - 1], odom[after]
        if s1 - s0 > 1e8:
            return None
        w = (t - s0) / max(s1 - s0, 1)
        T = np.eye(4)
        T[:3, 3] = (1 - w) * T0[:3, 3] + w * T1[:3, 3]
        T[:3, :3] = Slerp([0, 1], Rotation.from_matrix([T0[:3, :3], T1[:3, :3]]))([w]).as_matrix()[0]
        return T

    def take(stamp, lidar_frame, edges):
        T_l = pose_at(stamp)
        if T_l is None:
            return
        for c in cams:
            near = [m for m in images[c] if abs(m[0] - stamp) <= 1e8]
            if not near or c not in K:
                continue
            s, frame_id, jpeg = min(near, key=lambda m: abs(m[0] - stamp))
            T_c = pose_at(s)
            if T_c is None:
                continue
            P = np.linalg.inv(T_c) @ T_l @ tf.lookup('base_link', lidar_frame)  # LiDAR -> base at image time
            pts = P[:3, :3] @ edges + P[:3, 3:4]
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            scale = WIDTH / img.shape[1]
            img = cv2.resize(img, (WIDTH, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            e = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0), 50, 150)
            dist = cv2.distanceTransform(255 - e, cv2.DIST_L2, 3)
            size[c] = (scale, frame_id)
            samples[c].append((pts, dist, img))

    with open(args.bag, 'rb') as fh:
        wanted = sorted(set(cams) | set(infos) | {LIDAR, ODOM, '/tf_static'})
        for _, ch, _, msg in make_reader(fh, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                topics=wanted, log_time_order=True):
            t = ch.topic
            if t == '/tf_static':
                for tr in msg.transforms:
                    tf.edges[tr.child_frame_id.lstrip('/')] = (tr.header.frame_id.lstrip('/'),
                                                              tf_matrix(tr.transform.translation, tr.transform.rotation))
            elif t == ODOM:
                p = msg.pose.pose
                odom.append((stamp_ns(msg.header), tf_matrix(p.position, p.orientation)))
            elif t in images:
                images[t].append((stamp_ns(msg.header), msg.header.frame_id.lstrip('/'), bytes(msg.data)))
            elif t in infos:
                K[infos[t]] = np.array(msg.p, np.float64).reshape(3, 4)[:, :3]
            elif t == LIDAR:
                s = stamp_ns(msg.header)
                t0 = t0 or s
                moving = len(odom) > 10 and np.linalg.norm(odom[-1][1][:3, 3] - odom[-10][1][:3, 3]) > 0.3
                if (s - t0) / 1e9 >= args.start and moving and (last is None or s - last >= 1e9):
                    last = s
                    pending.append((s, msg.header.frame_id.lstrip('/'), lidar_edges(msg)))
            newest = max((d[-1][0] for d in images.values() if d), default=0)
            while pending and newest > pending[0][0] + 1e8:
                take(*pending.popleft())
            if min((len(v) for v in samples.values()), default=0) >= args.frames:
                break
    print(f'LiDAR {LIDAR}; ' + ', '.join(f'{c.split("/")[1]}: {len(v)} frames' for c, v in samples.items()), flush=True)

    def project(c, corr, pts):
        scale, frame_id = size[c]
        Kc = K[c].copy()
        Kc[:2] *= scale
        T = np.linalg.inv(tf.lookup('base_link', frame_id) @ corr)
        q = T[:3, :3] @ pts + T[:3, 3:4]
        q = q[:, q[2] > 1.0]
        return (Kc @ (q / q[2]))[:2].round().astype(int), q[2]

    def score(c, corr):
        tot = n = 0.0
        for pts, dist, _ in samples[c]:
            uv, _ = project(c, corr, pts)
            h, w = dist.shape
            ok = (uv[0] >= 0) & (uv[0] < w) & (uv[1] >= 0) & (uv[1] < h)
            tot += np.exp(-(dist[uv[1, ok], uv[0, ok]] / 3.0) ** 2).sum()  # 1 on an image edge, ~0 beyond 6 px
            n += ok.sum()
        return tot / max(n, 1)

    calib, lines = {}, []
    for c in cams:
        if not samples[c]:
            continue
        best, best_s = np.zeros(3), score(c, np.eye(4))
        base = best_s
        for step, span in ((0.5, 3.0), (0.1, 0.5)):  # coarse-to-fine over pitch, yaw, roll
            g = np.arange(-span, span + 1e-9, step)
            centre = best.copy()
            for d in np.array(np.meshgrid(g, g, g)).reshape(3, -1).T:
                s = score(c, rot(centre + d))
                if s > best_s:
                    best, best_s = centre + d, s
        frame_id = size[c][1]
        calib[frame_id] = rot(best).tolist()
        lines.append(f'{c.split("/")[1]:16s} edge alignment {base:.3f} -> {best_s:.3f}   correction pitch {best[0]:+.1f} '
                     f'yaw {best[1]:+.1f} roll {best[2]:+.1f} deg (about camera x right, y down, z forward)')
        print(lines[-1], flush=True)
        pts, _, img = samples[c][len(samples[c]) // 2]  # overlay: /tf_static mount | corrected
        panels = []
        for corr, title in ((np.eye(4), '/tf_static'), (rot(best), 'corrected')):
            uv, z = project(c, corr, pts)
            pan = img.copy()
            col = cv2.applyColorMap(np.clip(255 - z * 255 / 30, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_JET)[:, 0]
            for (u, v), cc in zip(uv.T, col):
                if 0 <= u < pan.shape[1] and 0 <= v < pan.shape[0]:
                    cv2.circle(pan, (int(u), int(v)), 2, tuple(int(x) for x in cc), -1)
            cv2.putText(pan, f'{c.split("/")[1]} {title}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            panels.append(pan)
        cv2.imwrite(str(out / f'overlay_{c.split("/")[1]}.jpg'), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / 'calibration.json').write_text(json.dumps(calib, indent=1))
    (out / 'report.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
