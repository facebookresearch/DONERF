# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def generate_ray_directions(w, h, fov, focal):
    x_dist = np.tan(fov / 2) * focal
    y_dist = x_dist * (h / w)
    x_dist_pp = x_dist / (w / 2)
    y_dist_pp = y_dist / (h / 2)

    start = np.array([-(x_dist - x_dist_pp/2), -(y_dist - y_dist_pp/2), focal])
    ray_d = np.repeat(start[None], repeats=w * h, axis=0).reshape((h, w, -1))
    w_range = np.repeat(np.arange(0, w)[None], repeats=h, axis=0)
    h_range = np.repeat(np.arange(0, h)[None], repeats=w, axis=0).T
    ray_d[:, :, 0] += x_dist_pp * w_range
    ray_d[:, :, 1] += y_dist_pp * h_range

    dirs = ray_d / np.tile(np.linalg.norm(ray_d, axis=2)[:, :, None], (1, 1, 3))
    dirs[:, :, 1] *= -1.
    dirs[:, :, 2] *= -1.
    return dirs
