import numpy as np


def shift_centroid_to_origin(points):
    """Move the input points so that their centroid (mid point of bounding box) is at origin.

    Args:
        points (np.array(-1,3)): set of points from which the centroid is to be computed.

    Returns:
        np.array(-1,3): set of points with centroid shifted to the origin.
    """
    num_points = points.shape[0]
    min_x = min(points[:, 0])
    min_y = min(points[:, 1])
    min_z = min(points[:, 2])
    max_x = max(points[:, 0])
    max_y = max(points[:, 1])
    max_z = max(points[:, 2])
    x_ = np.full(shape=num_points, fill_value=(min_x + max_x) / 2.0)
    y_ = np.full(shape=num_points, fill_value=(min_y + max_y) / 2.0)
    z_ = np.full(shape=num_points, fill_value=(min_z + max_z) / 2.0)
    points[:, 0] = points[:, 0] - x_
    points[:, 1] = points[:, 1] - y_
    points[:, 2] = points[:, 2] - z_
    return points
