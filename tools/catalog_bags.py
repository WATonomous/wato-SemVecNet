#!/usr/bin/env python3
"""Catalog remote MCAP bags without downloading them.

The script lists ``gdrive:`` recursively, then reads each MCAP's 37-byte
footer and the channel/statistics records at the beginning of its MCAP
summary with ``rclone cat --offset --count``.  It never reads message data or
chunks.

    export RCLONE_CONFIG=/home/arch/Desktop/code/wato-SemVecNet/rclone.conf
    python tools/catalog_bags.py
"""
import argparse
import datetime as dt
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import struct
import subprocess
import sys


REPO = Path(__file__).resolve().parent.parent
if (REPO / 'rclone.conf').is_file():
    os.environ.setdefault('RCLONE_CONFIG', str(REPO / 'rclone.conf'))

DEFAULT_OUTPUT = Path('/home/arch/Desktop/data/bag_catalog.json')
FOOTER_SIZE = 37
MAGIC = b'\x89MCAP0\r\n'
OP_CHANNEL = 0x04
OP_STATISTICS = 0x0B
WATERLOO_LAT = 43.47
WATERLOO_LON = -80.54


def solar_elevation(stamp_ns, lat=WATERLOO_LAT, lon=WATERLOO_LON):
    """Same NOAA approximation and thresholds used by build_finetune_dataset."""
    t = dt.datetime.fromtimestamp(stamp_ns / 1e9, dt.timezone.utc)
    g = 2 * math.pi / 365 * (t.timetuple().tm_yday - 1 + (t.hour + t.minute / 60 - 12) / 24)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                    - 0.040849 * math.sin(2 * g))
    ha = math.radians(((t.hour * 60 + t.minute + t.second / 60) + eqt + 4 * lon) / 4 - 180)
    la = math.radians(lat)
    return math.degrees(math.asin(math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)))


def light_at(stamp_ns):
    if not stamp_ns:
        return None
    elevation = solar_elevation(stamp_ns)
    return 'day' if elevation > 6 else 'dusk' if elevation > -6 else 'night'


def rclone_bytes(remote_path, offset, count):
    """Fetch exactly a remote byte range; rclone exits nonzero on a bad range."""
    return subprocess.check_output(
        ['rclone', 'cat', remote_path, '--offset', str(offset), '--count', str(count)],
        stderr=subprocess.PIPE,
    )


class SummaryReader:
    """Small cached range reader whose callers only address the summary."""

    block_size = 1024 * 1024

    def __init__(self, remote_path, start, end):
        self.remote_path = remote_path
        self.start = start
        self.end = end
        self.blocks = {}

    def read(self, offset, count):
        if count < 0 or offset < self.start or offset + count > self.end:
            raise ValueError('range is outside the MCAP summary')
        result = bytearray()
        while count:
            block_start = self.start + (offset - self.start) // self.block_size * self.block_size
            block_offset = offset - block_start
            if block_start not in self.blocks:
                self.blocks[block_start] = rclone_bytes(
                    self.remote_path, block_start, min(self.block_size, self.end - block_start))
            block = self.blocks[block_start]
            piece = block[block_offset:block_offset + count]
            if not piece:
                raise ValueError(f'unexpected end of range at offset {offset}')
            result.extend(piece)
            offset += len(piece)
            count -= len(piece)
        return bytes(result)


def parse_channel_topic(body):
    """Return the topic from an MCAP Channel record's leading fields."""
    if len(body) < 8:
        raise ValueError('short Channel record')
    topic_length = struct.unpack_from('<I', body, 4)[0]
    end = 8 + topic_length
    if end > len(body):
        raise ValueError('truncated Channel topic')
    return body[8:end].decode('utf-8', errors='replace')


def summary_info(remote_path, size):
    """Parse channels and statistics without entering MCAP message/chunk data."""
    if size < FOOTER_SIZE:
        raise ValueError('file is smaller than an MCAP footer')
    footer_offset = size - FOOTER_SIZE
    footer = rclone_bytes(remote_path, footer_offset, FOOTER_SIZE)
    if len(footer) != FOOTER_SIZE or footer[0] != 0x02 or footer[-8:] != MAGIC:
        raise ValueError('not a valid MCAP footer')
    summary_start, summary_offset_start = struct.unpack_from('<QQ', footer, 9)
    if not 0 < summary_start < footer_offset or not summary_start <= summary_offset_start <= footer_offset:
        raise ValueError('MCAP has no valid summary')

    # MCAP summaries order Schema records, Channel records, then Statistics.
    # Stop at Statistics: all later summary records are indexes and can be huge.
    reader = SummaryReader(remote_path, summary_start, summary_offset_start)
    offset = summary_start
    topics = []
    statistics = None
    while offset + 9 <= summary_offset_start:
        header = reader.read(offset, 9)
        opcode, length = struct.unpack('<BQ', header)
        body_offset = offset + 9
        record_end = body_offset + length
        if record_end > summary_offset_start:
            raise ValueError('summary record extends outside summary section')
        if opcode == OP_CHANNEL:
            # The topic is immediately after the two uint16 IDs and uint32 length.
            prefix = reader.read(body_offset, min(length, 8))
            if len(prefix) < 8:
                raise ValueError('short Channel record')
            topic_length = struct.unpack_from('<I', prefix, 4)[0]
            if 8 + topic_length > length:
                raise ValueError('invalid Channel topic length')
            topic = parse_channel_topic(reader.read(body_offset, 8 + topic_length))
            topics.append(topic)
        elif opcode == OP_STATISTICS:
            if length < 42:
                raise ValueError('short Statistics record')
            stats = reader.read(body_offset, 42)
            # See MCAP Statistics: start/end log times are at byte offsets 26/34.
            start_ns, end_ns = struct.unpack_from('<QQ', stats, 26)
            statistics = {'message_start_time_ns': start_ns, 'message_end_time_ns': end_ns}
            break
        offset = record_end
    if statistics is None:
        raise ValueError('summary has no Statistics record')
    return sorted(set(topics)), statistics


def camera_roots(topics):
    roots = set()
    for topic in topics:
        for component in topic.strip('/').split('/'):
            if component.startswith('camera_pano_'):
                roots.add(component)
                break
    return sorted(roots)


def gps_topics(topics):
    known = {'/novatel/oem7/odom', '/novatel/oem7/inspva'}
    return sorted(topic for topic in topics if topic in known or 'gnss' in topic.lower())


def snow_status(start_ns):
    if not start_ns:
        return 'snow_unknown'
    month = dt.datetime.fromtimestamp(start_ns / 1e9, dt.timezone.utc).month
    return 'possible' if month in (11, 12, 1, 2, 3) else 'snow_unknown'


def catalog_one(entry, remote):
    path = entry['Path']
    remote_path = remote.rstrip(':') + ':' + path
    record = {
        'path': path,
        'size_bytes': entry.get('Size'),
        'topics': [],
        'camera_topics': [],
        'camera_count': 0,
        'gps_topics': [],
        'has_gps': False,
        'message_start_time_ns': None,
        'message_start_utc': None,
        'duration_seconds': None,
        'light': None,
        'snow': 'snow_unknown',
        'usable': False,
    }
    try:
        topics, stats = summary_info(remote_path, entry['Size'])
        start_ns, end_ns = stats['message_start_time_ns'], stats['message_end_time_ns']
        cameras = camera_roots(topics)
        gps = gps_topics(topics)
        record.update(
            topics=topics,
            camera_topics=cameras,
            camera_count=len(cameras),
            gps_topics=gps,
            has_gps=bool(gps),
            message_start_time_ns=start_ns,
            message_start_utc=dt.datetime.fromtimestamp(start_ns / 1e9, dt.timezone.utc).isoformat(),
            duration_seconds=round((end_ns - start_ns) / 1e9, 3),
            light=light_at(start_ns),
            snow=snow_status(start_ns),
        )
        record['usable'] = record['has_gps'] and record['camera_count'] in (6, 8)
    except (OSError, ValueError, subprocess.CalledProcessError, struct.error) as exc:
        record['error'] = str(exc)
    return record


def print_table(records):
    rows = []
    for item in records:
        status = 'yes' if item['usable'] else 'no'
        gps = 'yes' if item['has_gps'] else 'no'
        duration = '-' if item['duration_seconds'] is None else f'{item["duration_seconds"]:.0f}s'
        start = item['message_start_utc'] or '-'
        if len(start) > 19:
            start = start[:19] + 'Z'
        rows.append([item['path'], str(item['camera_count']), gps, duration, start, item['light'] or '-',
                     item['snow'], status])
    headers = ['path', 'cams', 'gps', 'duration', 'start (UTC)', 'light', 'snow', 'usable']
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], min(len(cell), 72 if i == 0 else len(cell)))
    print('  '.join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print('  '.join('-' * width for width in widths))
    for row in rows:
        row[0] = row[0] if len(row[0]) <= widths[0] else '...' + row[0][-(widths[0] - 3):]
        print('  '.join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--remote', default='gdrive:', help='rclone remote rooted at the Bags Drive folder')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--workers', type=int, default=4, help='simultaneous remote summary reads')
    args = ap.parse_args()
    if args.workers < 1:
        ap.error('--workers must be at least 1')

    try:
        listing = json.loads(subprocess.check_output(['rclone', 'lsjson', '-R', args.remote], text=True))
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        sys.exit(f'Could not list {args.remote}: {exc}')
    bags = sorted(
        (item for item in listing
         if not item.get('IsDir') and item.get('Path', '').lower().endswith('.mcap')
         and 'nuscenes' not in {part.lower() for part in Path(item['Path']).parts}),
        key=lambda item: item['Path'],
    )
    print(f'Found {len(bags)} non-NuScenes MCAP bag(s); reading footers and summaries only...', file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(catalog_one, item, args.remote) for item in bags]
        records = [future.result() for future in as_completed(futures)]
    records.sort(key=lambda item: item['path'])
    result = {
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'remote': args.remote,
        'excluded_path_component': 'NuScenes',
        'waterloo_lat_lon': [WATERLOO_LAT, WATERLOO_LON],
        'bags': records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print_table(records)
    print(f'\nWrote {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
