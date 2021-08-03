# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
import json

import numpy as np
import transforms3d as t3d


class Camera:
    @classmethod
    def calc_positions(cls, **kwargs) -> torch.tensor:
        pass


class CenteredCamera:
    @classmethod
    def calc_positions(cls, config, **kwargs):
        matrices = []
        for angle in np.arange(0, 2 * np.pi, 2 * np.pi / config.videoFrames):
            T = np.eye(4)
            T[:3, 3] = np.array(config.camCenter)
            T[2, 3] += config.camRadius
            R = np.eye(4)
            R[:3, :3] = t3d.euler.euler2mat(np.sin(angle) * np.deg2rad(config.camRightAngle),
                                            np.cos(angle) * np.deg2rad(config.camUpAngle), 0)
            M = R @ T
            matrices.append(M[None])
        return np.concatenate(matrices, axis=0)


class RotatingCamera:
    @classmethod
    # This camera simply rotates without movement.
    def calc_positions(cls, config, **kwargs):
        matrices = []
        for angle in np.arange(0, 2 * np.pi, 2 * np.pi / config.videoFrames):
            T = np.eye(4)
            T[:3, 3] = np.array(config.camCenter)
            T[2, 3] += config.camRadius
            T[:3, :3] = t3d.euler.euler2mat(np.sin(angle) * np.deg2rad(config.camRightAngle),
                                            np.cos(angle) * np.deg2rad(config.camUpAngle), 0)
            matrices.append(T[None])
        return np.concatenate(matrices, axis=0)


class TranslatingCamera:
    @classmethod
    # This camera simply translates without rotation.
    def calc_positions(cls, config, **kwargs):
        matrices = []
        for step in np.arange(-1.0, 1.0, 2.0 / config.videoFrames):
            T = np.eye(4)
            T[:3, 3] = np.array(config.camCenter)
            T[2, 3] += config.camRadius
            T[0:3, 3] += np.array(config.movementVector) * step
            matrices.append(T[None])
        return np.concatenate(matrices, axis=0)


class ViewCellForwardCamera:
    @classmethod
    # This camera simply translates without rotation, based on a view cell starting position along a given axis.
    def calc_positions(cls, config, **kwargs):
        matrices = []

        data = kwargs.get('data', None)

        view_cell_center = np.array(data.view.view_cell_center)
        view_cell_size = np.array(data.view.view_cell_size)

        for step in np.arange(0, 1.0, 1.0 / config.videoFrames):
            T = np.eye(4)

            # NOTE: this is just hardcoded from forest to get the initial orientation, since
            #       we didn't have initial orientation in this dataset...
            T[1, 0:3] = np.array([0, 0, -1])
            T[2, 0:3] = np.array([0, 1, 0])

            T[:3, 3] = view_cell_center - (view_cell_size / 2) * np.array(config.movementVector)
            T[0:3, 3] += np.array(config.movementVector) * step * view_cell_size
            matrices.append(T[None])

        return np.concatenate(matrices, axis=0)


class PredefinedCamera:
    @classmethod
    # This camera loaded form a file
    def calc_positions(cls, config, **kwargs):
        frames = 0 if not config.videoFrames else config.videoFrames

        return cls.import_camera_path(config.data, config.camPath, frames)

    @classmethod
    def import_camera_path(cls, path, file_name, num_frames=-1):
        transforms = None
        with open(os.path.join(path, f"{file_name}.json"), "r") as f:
            file = json.load(f)

            for frame_idx, frame in enumerate(file["frames"]):
                pose = np.array(frame["transform_matrix"]).astype(np.float32)

                if transforms is None:
                    transforms = np.zeros((len(file["frames"]), pose.shape[0], pose.shape[1]), dtype=np.float32)

                transforms[frame_idx] = pose

        if 0 < num_frames < len(transforms):
            transforms = transforms[:num_frames]

        return transforms
