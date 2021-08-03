# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

import numpy as np


class LogTransform:
    @classmethod
    def from_world(cls, depth, depth_range):
        min_d = depth_range[0]
        max_d = depth_range[1]

        max_v = max_d - min_d

        depth -= min_d  # 0..max_v

        if isinstance(depth, torch.Tensor):
            non_lin_depth = torch.log(depth + 1.)  # no log with custom base in pytorch, so using this
            non_lin_depth = non_lin_depth / math.log(max_v + 1)
        elif isinstance(depth, np.ndarray):
            non_lin_depth = np.log(depth + 1.)  # no log with custom base in numpy, so using this
            non_lin_depth = non_lin_depth / math.log(max_v + 1)
        else:
            non_lin_depth = math.log(depth + 1., max_v + 1)

        return non_lin_depth

    @classmethod
    def to_world(cls, depth, depth_range):
        min_d = depth_range[0]
        max_d = depth_range[1]

        max_v = max_d - min_d

        world_depth = (max_v + 1) ** depth
        world_depth -= 1.
        world_depth += depth_range[0]

        return world_depth


class LinearTransform:
    @classmethod
    def from_world(cls, depth, depth_range):
        return (depth - depth_range[0]) / (depth_range[1] - depth_range[0])

    @classmethod
    def to_world(cls, depth, depth_range):
        return depth * (depth_range[1] - depth_range[0]) + depth_range[0]
