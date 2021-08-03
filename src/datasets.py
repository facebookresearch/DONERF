# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import sys
import json
import torch
import imageio
import importlib

import numpy as np
import util.depth_transformations as depth_transforms

from tqdm import trange
from torch.utils.data import Dataset, get_worker_info
from util.raygeneration import generate_ray_directions


class DatasetKeyConstants:
    color_image_full = "ColorImageFull"
    color_image_samples = "ColorImageSamples"
    depth_image_full = "DepthImageFull"
    depth_image_samples = "DepthImageSamples"
    image_sample_indices = "ImageSampleIndices"
    image_pose = "ImagePose"
    image_rotation = "ImageRotation"
    ray_directions = "RayDirections"
    ray_directions_samples = "RayDirectionsSamples"
    batch_input_dir = "BatchInputDir"
    train_target = "TrainTarget"


def create_sample_wrapper(sample_data, train_config, single=False):
    batch_input_dirs = []
    train_targets = []

    for i in range(len(train_config.f_in)):
        batch_input_dirs.append(sample_data[f"{DatasetKeyConstants.batch_input_dir}_{i}"])

        # if we are not training, we have no targets
        target_string = f"{DatasetKeyConstants.train_target}_{i}"
        if target_string in sample_data:
            train_targets.append(sample_data[target_string])

    if train_config.copy_to_gpu:
        for idx in range(len(batch_input_dirs)):
            for key in batch_input_dirs[idx]:
                if isinstance(batch_input_dirs[idx][key], torch.Tensor):
                    batch_input_dirs[idx][key] = batch_input_dirs[idx][key].to(train_config.config_file.device,
                                                                               non_blocking=True)

        for idx in range(len(train_targets)):
            if isinstance(train_targets[idx], torch.Tensor):
                train_targets[idx] = train_targets[idx].to(train_config.config_file.device, non_blocking=True)

    return SampleDataWrapper(batch_input_dirs, train_targets, single)


class SampleDataWrapper:
    def __init__(self, batch_input_dirs, train_targets, single):
        self.batch_input_dirs = batch_input_dirs
        self.train_targets = train_targets
        self.single = single

    def get_batch_input(self, index):
        return self.batch_input_dirs[index]

    def get_train_target(self, index):
        return self.train_targets[index]

    def batches(self, batch_size):
        n_samples = -1

        samples_location = 0 if self.single else 1

        if DatasetKeyConstants.ray_directions_samples in self.batch_input_dirs[0]:
            n_samples = self.batch_input_dirs[0][DatasetKeyConstants.ray_directions_samples].shape[samples_location]
        elif DatasetKeyConstants.color_image_samples in self.batch_input_dirs[0]:
            n_samples = self.batch_input_dirs[0][DatasetKeyConstants.color_image_samples].shape[samples_location]
        elif DatasetKeyConstants.depth_image_samples in self.batch_input_dirs[0]:
            n_samples = self.batch_input_dirs[0][DatasetKeyConstants.depth_image_samples].shape[samples_location]

        if n_samples == -1:
            print("ERROR: unable to batch sample data!")

        for batch0 in range(0, n_samples, batch_size):
            batch_input_dirs = []
            train_targets = []

            for idx in range(len(self.batch_input_dirs)):
                inner_dir = {}

                for key in self.batch_input_dirs[idx]:
                    if key == DatasetKeyConstants.color_image_samples or \
                            key == DatasetKeyConstants.depth_image_samples or \
                            key == DatasetKeyConstants.ray_directions_samples:
                        if self.single:
                            inner_dir[key] = self.batch_input_dirs[idx][key][None, batch0:batch0 + batch_size, :]
                        else:
                            inner_dir[key] = self.batch_input_dirs[idx][key][:, batch0:batch0 + batch_size, :]
                    else:
                        if self.single:
                            inner_dir[key] = self.batch_input_dirs[idx][key][None]
                        else:
                            inner_dir[key] = self.batch_input_dirs[idx][key]

                batch_input_dirs.append(inner_dir)

            for idx in range(len(self.train_targets)):
                if self.single:
                    train_targets.append(self.train_targets[idx][batch0:batch0 + batch_size])
                else:
                    train_targets.append(self.train_targets[idx][0, batch0:batch0 + batch_size])

            yield SampleDataWrapper(batch_input_dirs, train_targets, False)


class View:
    def __init__(self):
        self.fov = 0.0
        self.focal = 0.0
        self.camera_scale = 1.0
        self.view_cell_center = [0, 0, 0]
        self.view_cell_size = [0, 0, 0]


class DatasetInfo:
    def __init__(self, config, train_config):
        self.config = config
        self.train_config = train_config
        self.dataset_path = config.data
        self.view = View()
        self.scale = config.scale

        from features import SpherePosDir
        self.use_warped_depth_range = isinstance(train_config.f_in[0], SpherePosDir)

        # read in dataset specific .json
        with open(os.path.join(self.dataset_path, "dataset_info.json"), "r") as f:
            dataset_info = json.load(f)
        self.view.view_cell_center = dataset_info["view_cell_center"]
        self.view.view_cell_size = dataset_info["view_cell_size"]
        self.view.camera_scale = 1.
        if "camera_scale" in dataset_info:
            self.view.camera_scale = float(dataset_info["camera_scale"])

        self.w, self.h = dataset_info["resolution"][0], dataset_info["resolution"][1]
        if self.scale > 1:
            self.w = self.w // self.scale
            self.h = self.h // self.scale

        self.train_config.h = self.h
        self.train_config.w = self.w

        self.view.fov = float(dataset_info["camera_angle_x"])
        self.view.focal = float(.5 * self.w / np.tan(.5 * self.view.fov))

        # vertically flip loaded depth files
        self.flip_depth = dataset_info["flip_depth"]

        # adjustments if depth is based on distance to camera plane, and not distance to camera origin
        self.depth_distance_adjustment = dataset_info["depth_distance_adjustment"]

        if "depth_ignore" not in dataset_info or "depth_range" not in dataset_info or \
                "depth_range_warped_log" not in dataset_info or \
                "depth_range_warped_lin" not in dataset_info:
            print("error: necessary depth range information not found in 'dataset_info.json'")
            sys.exit(-1)

        self.depth_ignore = float(dataset_info["depth_ignore"])

        self.depth_range = [float(dataset_info["depth_range"][0]), float(dataset_info["depth_range"][1])]

        self.depth_max = self.depth_range[1]

        if config.depthTransform == "linear":
            self.depth_transform = depth_transforms.LinearTransform
            self.depth_range_warped = [float(dataset_info["depth_range_warped_lin"][0]),
                                       float(dataset_info["depth_range_warped_lin"][1])]
        elif config.depthTransform == "log":
            self.depth_transform = depth_transforms.LogTransform
            self.depth_range_warped = [float(dataset_info["depth_range_warped_log"][0]),
                                       float(dataset_info["depth_range_warped_log"][1])]


class ViewCellDataset(Dataset):
    def __init__(self, config, train_config, dataset_info, set_name="train", num_samples=2048):
        self.config = config
        self.train_config = train_config
        self.dataset_path = config.data
        self.set_name = set_name
        self.num_samples = num_samples
        self.image_filenames = []
        self.depth_filenames = None
        self.view = dataset_info.view
        self.scale = dataset_info.scale
        self.flip_depth = dataset_info.flip_depth
        self.depth_distance_adjustment = dataset_info.depth_distance_adjustment
        self.depth_ignore = dataset_info.depth_ignore
        self.depth_max = dataset_info.depth_max
        self.depth_range = dataset_info.depth_range
        self.depth_range_warped = dataset_info.depth_range_warped
        self.depth_transform = dataset_info.depth_transform
        self.w = dataset_info.w
        self.h = dataset_info.h
        self.num_items = 0
        self.device = config.device
        self.is_inference = False  # if True, does not give training targets
        self.full_images = False  # if True, gives sample data for whole images
        self.poses = None
        self.rotations = None
        self.directions = None

        self.base_ray_z = np.abs(
            generate_ray_directions(self.w, self.h, self.view.fov, self.view.focal)[:, :, 2]).astype(np.float32)

        if set_name == "test":
            # we do not set is_inference to True, because we still want training targets for images
            self.full_images = True
        elif set_name == "vid":
            self.is_inference = True
            self.full_images = True

    def preprocess_pos_and_dir(self, transforms):
        self.poses = transforms[:, :3, 3:].reshape(-1, 3)
        self.poses = torch.tensor(self.poses, device=self.device, dtype=torch.float32)

        self.rotations = torch.tensor(transforms[:, :3, :3], device=self.device, dtype=torch.float32)

        npdirs = generate_ray_directions(self.w, self.h, self.view.fov, self.view.focal)
        self.directions = torch.tensor(npdirs.flatten().reshape(-1, 3), device=self.device, dtype=torch.float32)

    def scale_image(self, image):
        return cv2.resize(image, (image.shape[0] // self.scale,
                                  image.shape[1] // self.scale), interpolation=cv2.INTER_AREA)

    def load_color_image(self, file_name):
        color_image = imageio.imread(file_name).astype(np.float32)
        if self.scale > 1:
            color_image = self.scale_image(color_image)

        color_image = color_image / 255.
        return color_image[:, :, :3]

    def load_depth_image(self, file_name):
        np_file = np.load(file_name)
        depth_image = np_file["depth"] if "depth" in np_file.files else np_file[np_file.files[0]]
        depth_image = depth_image.astype(np.float32)
        depth_image = np.resize(depth_image, (self.h * self.scale, self.w * self.scale))

        if self.flip_depth:
            depth_image = np.flip(depth_image, 0)

        depth_only_max = depth_image.copy()
        depth_only_max[depth_only_max != self.depth_ignore] = 0
        depth_only_max = self.scale_image(depth_only_max)

        if self.scale > 1:
            if self.config.scaleInterpolation == "area":
                depth_image = self.scale_image(depth_image)
            elif self.config.scaleInterpolation == "median":
                # we first need to create a "stacked" version of the image where we have each pixel in one of the stacks
                stacked_depths = []
                for i in range(self.scale):
                    for j in range(self.scale):
                        stacked_depths.append(depth_image[i::self.scale, j::self.scale])

                depth_sorted = np.sort(np.dstack(stacked_depths), -1)
                # Take the element that is just smaller than the median
                depth_image = depth_sorted[:, :, self.scale - 1]
            else:
                depth_image = depth_image[0::self.scale, 0::self.scale]

        depth_image[depth_only_max != 0] = self.depth_ignore

        if self.depth_distance_adjustment:
            depth_image = depth_image / self.base_ray_z[:, :]

        depth_image = (depth_image - self.depth_range[0]) / (self.depth_range[1] - self.depth_range[0])

        depth_image = self.depth_transform.from_world(
            depth_transforms.LinearTransform.to_world(depth_image, self.depth_range), self.depth_range)
        depth_image[depth_only_max != 0] = 1.

        depth_image = depth_image.reshape(1, self.h, self.w, 1)

        return depth_image

    def get_random_sample_indices(self, device="cpu"):
        if not self.full_images:
            rand_pixels_2d = self.train_config.pixel_idx_sequence_gen.get_discrete_tensor_subset(self.num_samples,
                                                                                                 device="cpu",
                                                                                                 minv=0,
                                                                                                 maxv=torch.tensor(
                                                                                                     [self.h, self.w],
                                                                                                     dtype=torch.long))

            random_sample_indices = rand_pixels_2d[:, 0] + self.h * rand_pixels_2d[:, 1]
        else:
            random_sample_indices = torch.tensor([i for i in range(self.w * self.h)], device=device)

        return random_sample_indices.to(device)

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        pass


# worker init function for OnThFlyViewCellDataset
def worker_offset_sequence(worker_id):
    worker_info = get_worker_info()
    # we set the offset such that all workers start at different offsets
    offset = int((worker_info.dataset.h * worker_info.dataset.w / worker_info.num_workers) * worker_id)
    worker_info.dataset.train_config.pixel_idx_sequence_gen.set_offset(offset)


class OnTheFlyViewCellDataset(ViewCellDataset):
    def __init__(self, config, train_config, dataset_info, set_name="train", num_samples=2048):
        super(OnTheFlyViewCellDataset, self).__init__(config, train_config, dataset_info, set_name, num_samples)

        # on the fly loading only works on CPU
        self.device = "cpu"

        # read in transformation .json
        with open(os.path.join(self.dataset_path, f"transforms_{set_name}.json"), "r") as f:
            json_data = json.load(f)

        self.num_items = len(json_data["frames"])
        transforms = None

        for frame_idx, frame in enumerate(json_data["frames"]):
            # store image file paths
            file_path = os.path.join(self.dataset_path, frame["file_path"])
            file_name = file_path + ".png"
            self.image_filenames.append(file_name)

            # store depth file paths if present
            depth_name = file_path + "_depth.npz"
            if self.depth_filenames is None and os.path.exists(depth_name):
                self.depth_filenames = [depth_name]
            elif os.path.exists(depth_name):
                self.depth_filenames.append(depth_name)

            pose = np.array(frame["transform_matrix"]).astype(np.float32)

            # store transformations for all images
            if transforms is None:
                transforms = np.empty((self.num_items, pose.shape[0], pose.shape[1]), dtype=np.float32)
            transforms[frame_idx] = pose

        self.preprocess_pos_and_dir(transforms)

    def __getitem__(self, index):
        color_image = self.load_color_image(self.image_filenames[index])
        depth_image = None

        if self.depth_filenames is not None:
            depth_image = self.load_depth_image(self.depth_filenames[index])

        random_sample_indices = self.get_random_sample_indices("cpu")

        data_item = {DatasetKeyConstants.color_image_full: torch.tensor(color_image, device=self.device)[None],
                     DatasetKeyConstants.depth_image_full: torch.tensor(depth_image, device=self.device),
                     DatasetKeyConstants.image_sample_indices: random_sample_indices,
                     DatasetKeyConstants.image_pose: self.poses[index][None, :],
                     DatasetKeyConstants.image_rotation: self.rotations[index][None, :],
                     DatasetKeyConstants.ray_directions: self.directions}

        sample_dict = {}

        for feature_idx in range(len(self.train_config.f_in)):
            f_in = self.train_config.f_in[feature_idx]
            f_in.preprocess(data_item, self.device, self.config)

            # get output of prepare_batch, which is the input to later batch function call
            in_prepared_batch = f_in.prepare_batch(data_item, self.config)
            sample_dict[f"{DatasetKeyConstants.batch_input_dir}_{feature_idx}"] = in_prepared_batch

            if not self.is_inference:
                f_out = self.train_config.f_out[feature_idx]
                f_out.preprocess(data_item, self.device, self.config)

                # get output of prepare_batch
                out_prepared_batch = f_out.prepare_batch(data_item, self.config)

                train_target = f_out.batch(out_prepared_batch)
                sample_dict[f"{DatasetKeyConstants.train_target}_{feature_idx}"] = train_target

        return sample_dict


class FullyLoadedViewCellDataset(ViewCellDataset):
    def __init__(self, config, train_config, dataset_info, set_name="train", num_samples=2048):
        super(FullyLoadedViewCellDataset, self).__init__(config, train_config, dataset_info, set_name, num_samples)

        with open(os.path.join(self.dataset_path, f"transforms_{set_name}.json"), "r") as f:
            json_data = json.load(f)

        self.num_items = len(json_data["frames"])
        transforms = None
        self.color_images = None

        tqdm_range = trange(len(json_data["frames"]), desc=f"Loading dataset {set_name:5s}", leave=True)

        for frame_idx in tqdm_range:
            frame = json_data["frames"][frame_idx]
            file_path = os.path.join(self.dataset_path, frame["file_path"])
            file_name = file_path + ".png"

            color_image = self.load_color_image(file_name)
            depth_image = None
            pose = np.array(frame["transform_matrix"]).astype(np.float32)

            depth_name = file_path + "_depth.npz"
            if os.path.exists(depth_name):
                depth_image = self.load_depth_image(depth_name)

            if self.color_images is None:
                self.color_images = np.zeros((len(self), color_image.shape[0], color_image.shape[1],
                                              color_image.shape[2]), dtype=np.float32)
                transforms = np.zeros((len(self), pose.shape[0], pose.shape[1]), dtype=np.float32)

                if depth_image is not None:
                    self.depth_images = np.zeros((len(self), depth_image.shape[1], depth_image.shape[2], 1),
                                                 dtype=np.float32)

            self.color_images[frame_idx] = color_image
            transforms[frame_idx] = pose
            if depth_image is not None:
                self.depth_images[frame_idx] = depth_image[0]

        self.preprocess_pos_and_dir(transforms)

        self.color_images = torch.from_numpy(self.color_images).to(self.device)
        if self.depth_images is not None:
            self.depth_images = torch.from_numpy(self.depth_images).to(self.device)

        data = {DatasetKeyConstants.color_image_full: self.color_images,
                DatasetKeyConstants.depth_image_full: self.depth_images,
                DatasetKeyConstants.image_pose: self.poses,
                DatasetKeyConstants.image_rotation: self.rotations,
                DatasetKeyConstants.ray_directions: self.directions}

        # we now call preprocess on all features to perform necessary preprocess steps
        for feature_idx in range(len(self.train_config.f_in)):
            f_in = self.train_config.f_in[feature_idx]
            f_in.preprocess(data, self.device, self.config)

            self.depth_images = data[DatasetKeyConstants.depth_image_full]

            f_out = self.train_config.f_out[feature_idx]
            f_out.preprocess(data, self.device, self.config)

    def __getitem__(self, index):
        random_sample_indices = self.get_random_sample_indices(self.device)

        data_item = {DatasetKeyConstants.color_image_full: self.color_images[index][None, :],
                     DatasetKeyConstants.depth_image_full: self.depth_images[index][None, :],
                     DatasetKeyConstants.image_sample_indices: random_sample_indices,
                     DatasetKeyConstants.image_pose: self.poses[index][None, :],
                     DatasetKeyConstants.image_rotation: self.rotations[index][None, :],
                     DatasetKeyConstants.ray_directions: self.directions}

        sample_dict = {}

        for feature_idx in range(len(self.train_config.f_in)):
            f_in = self.train_config.f_in[feature_idx]

            # get output of prepare_batch, which is the input to later batch function call
            in_prepared_batch = f_in.prepare_batch(data_item, self.config)
            sample_dict[f"{DatasetKeyConstants.batch_input_dir}_{feature_idx}"] = in_prepared_batch

            if not self.is_inference:
                f_out = self.train_config.f_out[feature_idx]

                # get output of prepare_batch
                out_prepared_batch = f_out.prepare_batch(data_item, self.config)

                train_target = f_out.batch(out_prepared_batch)
                sample_dict[f"{DatasetKeyConstants.train_target}_{feature_idx}"] = train_target

        return sample_dict


class CameraViewCellDataset(ViewCellDataset):
    def __init__(self, config, train_config, dataset_info):
        super(CameraViewCellDataset, self).__init__(config, train_config, dataset_info, "vid", 2048)

        # Infer type and dynamically import based on config string.
        # This saves us the headache of maintaining if/else checks for the class type.
        camera_type = getattr(importlib.import_module("camera"), config.camType)
        transforms = camera_type.calc_positions(config)

        self.num_items = len(transforms)

        self.preprocess_pos_and_dir(transforms)

    def __getitem__(self, index):
        random_sample_indices = self.get_random_sample_indices(self.device)

        data_item = {DatasetKeyConstants.image_sample_indices: random_sample_indices,
                     DatasetKeyConstants.image_pose: self.poses[index][None, :],
                     DatasetKeyConstants.image_rotation: self.rotations[index][None, :],
                     DatasetKeyConstants.ray_directions: self.directions}

        sample_dict = {}

        for feature_idx in range(len(self.train_config.f_in)):
            f_in = self.train_config.f_in[feature_idx]

            # get output of prepare_batch, which is the input to later batch function call
            in_prepared_batch = f_in.prepare_batch(data_item, self.config)
            sample_dict[f"{DatasetKeyConstants.batch_input_dir}_{feature_idx}"] = in_prepared_batch

        return sample_dict
