"""Joint raster and vector augmentation for the exported 0.2 m Ring Road maps."""
import cv2
import numpy as np
from shapely.geometry import LineString, box


# build_finetune_dataset.draw_preview measures raster rows from the rear axle.
# Model coordinates use the virtual LiDAR 0.943713 m forward of that axle.
METRIC_TO_PIXEL = np.array([[5., 0., 75.], [0., -5., (30. - 0.943713) * 5.], [0., 0., 1.]])
PIXEL_TO_METRIC = np.linalg.inv(METRIC_TO_PIXEL)
PATCH = box(-15., -30., 15., 30.)


def clipped_lines(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == 'LineString' and geometry.length > 1e-6:
        yield np.asarray(geometry.coords, dtype=np.float32)
    elif geometry.geom_type in ('MultiLineString', 'GeometryCollection'):
        for part in geometry.geoms:
            yield from clipped_lines(part)


class RingRoadAffine:
    def __call__(self, results):
        angle = np.deg2rad(np.random.uniform(-10., 10.))
        flip = -1. if np.random.random() < 0.5 else 1.
        dx, dy = np.random.uniform(-1.5, 1.5, 2)
        c, s = np.cos(angle), np.sin(angle)
        affine = np.array([[flip * c, -s, dx], [flip * s, c, dy], [0., 0., 1.]])
        raster = results['sem_map']
        if raster.shape[:2] != (300, 150):
            raise ValueError('Ring Road augmentation expects a 300-row, 150-column map')
        pixel_affine = METRIC_TO_PIXEL @ affine @ PIXEL_TO_METRIC
        results['sem_map'] = cv2.warpAffine(raster, pixel_affine[:2], (150, 300),
                                            flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
        annotation = {}
        for name, lines in results['ann_info'].items():
            annotation[name] = []
            for points in lines:
                if len(points) >= 2:
                    moved = points @ affine[:2, :2].T + affine[:2, 2]
                    annotation[name].extend(clipped_lines(LineString(moved).intersection(PATCH)))
        # prepare_train_data reads this original input dict again after Collect3D.
        results['ann_info'] = annotation
        return results
