import math

import numpy as np


def direction_from_points(start_x, start_y, end_x, end_y, dirs=8):
    dx = start_x - end_x
    dy = start_y - end_y

    if dx == 0 and dy == 0:
        return 0

    angle_deg = math.degrees(math.atan2(dy, dx))
    angle_west_deg = (angle_deg - 180.0) % 360.0

    step = 360.0 / dirs
    return int(round(angle_west_deg / step)) % dirs


def bilinear_sample(K, x, y):
    h, w = K.shape

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    # outside kernel -> zero probability
    if x0 < 0 or y0 < 0 or x1 >= h or y1 >= w:
        return 0.0

    dx = x - x0
    dy = y - y0

    return (
        K[x0, y0] * (1 - dx) * (1 - dy) +
        K[x1, y0] * dx       * (1 - dy) +
        K[x0, y1] * (1 - dx) * dy +
        K[x1, y1] * dx       * dy
    )

def resample_kernel_to_grid(K_meter, cell_size, S):
    size = 2 * S + 1
    center = K_meter.shape[0] // 2

    K_grid = np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            dx_m = (i - S) * cell_size
            dy_m = (j - S) * cell_size

            x = center + dx_m
            y = center + dy_m

            K_grid[i, j] = bilinear_sample(K_meter, x, y)

    return K_grid / K_grid.sum()