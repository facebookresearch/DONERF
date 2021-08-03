# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

# Some code is adapted from nerf-pytorch https://github.com/yenchenlin/nerf-pytorch, which is licensed under MIT (see LICENSE_third_party.md in the root of this repository).

import torch

import numpy as np

from abc import ABC
from util.helper import tile
from importlib import import_module
from datasets import DatasetKeyConstants
from util.feature_encoding import FeatureEncoding
from nerf_raymarch_common import nerf_get_ray_dirs, nerf_raw2outputs, nerf_sample_pdf, \
    nerf_get_normalization_function, nerf_get_normalization_function_abbr


class FeatureSetKeyConstants:
    input_feature_batch = 'InputFeatureBatch'
    network_output = 'NetworkOutputBatch'
    postprocessed_network_output = 'PostProcessedNetworkOutput'
    input_feature_ray_directions = "InputFeatureRayDirections"
    input_feature_ray_origins = "InputFeatureRayOrigins"
    nerf_weights_output = "NeRFWeightsOutput"
    nerf_input_feature_z_vals = "NeRFInputFeatureZVals"
    nerf_estimated_depth = "NeRFOutputDepth"
    nerf_input_feature_ray_directions = input_feature_ray_directions
    nerf_input_feature_ray_origins = input_feature_ray_origins
    input_depth_groundtruth = "InputDepthGroundtruth"
    input_depth_groundtruth_world = "InputDepthGroundtruthWorld"
    input_depth_range = "InputDepthRange"
    input_depth = "InputDepth"
    quantization_max_weight = "QuantizationMaxWeight"
    quantized_weights = "QuantizedWeights"
    output_depth_map = "OutputDepthMap"


class FeatureSet(ABC):
    abbr = "Unknown"
    """number of input features for the first layer of the network"""
    n_feat = 0
    w = 0
    h = 0
    net_idx = -1

    def initialize(self, config, dataset_info, device):
        """creates all the necessary constant information of the feature"""
        pass

    def preprocess(self, data, device, config):
        """preprocesses data to accelerate batch creation"""
        pass

    def prepare_batch(self, data, config):
        """extracts and returns the necessary information as input for the batch function"""
        return None

    def batch(self, data, **kwargs) -> (torch.tensor, torch.tensor):
        """creates a single batch for learning using the images and samples in idx"""
        return None

    def postprocess(self, inference_dict, idx):
        """postprocesses the network outputs before applying the loss. required for e.g. NeRF (ray marching)"""
        inference_dict[FeatureSetKeyConstants.postprocessed_network_output] \
            = inference_dict[FeatureSetKeyConstants.network_output]

    def get_string(self):
        return self.abbr

    @classmethod
    def get_sets(cls, config, device):
        in_features = []
        out_features = []
        for i in range(len(config.inFeatures)):
            f_in = getattr(import_module("features"), config.inFeatures[i])(config=config, net_idx=i, device=device)
            f_out = getattr(import_module("features"), config.outFeatures[i])(config=config, net_idx=i, device=device)
            in_features.append(f_in)
            out_features.append(f_out)
        return in_features, out_features


class RGBARayMarch(FeatureSet):
    abbr = "RGBARayMarch"
    n_feat = 4

    def __init__(self, config=None, net_idx=-1, device="cpu", **kwargs):
        self.config = config
        self.device = device
        self.net_idx = net_idx

    def prepare_batch(self, data, config):
        ret_dict = {DatasetKeyConstants.color_image_full: data[DatasetKeyConstants.color_image_full][0],
                    DatasetKeyConstants.image_sample_indices: data[DatasetKeyConstants.image_sample_indices]}

        return ret_dict

    def batch(self, data, **kwargs) -> (torch.tensor, torch.tensor):
        image_indices = data[DatasetKeyConstants.image_sample_indices]

        color_image = data[DatasetKeyConstants.color_image_full]
        color_image = color_image.reshape(color_image.shape[0] * color_image.shape[1], color_image.shape[2])
        color_image = color_image[image_indices]

        return color_image


class ClassifiedDepth(FeatureSet):
    abbr = "CD"
    n_feat: int = 128
    window_size: int = 5
    d_window_size: int = 0
    center_id: int = 2
    ignore_depth_value: float = 1.
    d_kernel = None

    def __init__(self, config=None, net_idx=-1, device="cpu", **kwargs):
        if config.multiDepthFeatures:
            self.n_feat = config.multiDepthFeatures[net_idx]
        if config.multiDepthWindowSize:
            sizes = config.multiDepthWindowSize[net_idx].split(':')
            self.window_size = int(sizes[0])
            self.center_id = self.window_size // 2
            if len(sizes) > 1:
                self.d_window_size = int(sizes[1])
        if config.multiDepthIgnoreValue:
            self.ignore_depth_value = config.multiDepthIgnoreValue[net_idx]

        self.w = 0
        self.h = 0

        self.net_idx = net_idx
        self.device = device

        if self.d_window_size > 1:
            if self.d_window_size % 2 == 0:
                self.d_window_size += 1
            self.abbr = "CD-{}-{}-{}".format(self.n_feat, self.window_size, self.d_window_size)
            tri_base = np.linspace(0., 1., (self.d_window_size + 3) // 2)
            self.d_kernel = np.concatenate([tri_base[1:], tri_base[-2:0:-1]])
        else:
            self.abbr = "CD-{}-{}".format(self.n_feat, self.window_size)

        try:
            import disc_depth_multiclass as cuda_batch
        except ImportError:
            print("WARNING! - import of cuda kernels for 'disc_depth_multiclass' failed - falling back to PyTorch")
            cuda_batch = None

        self.cuda_batch = cuda_batch

    def initialize(self, config, dataset_info, device):
        self.w = dataset_info.w
        self.h = dataset_info.h

        if self.d_kernel is not None:
            dvc = "cpu" if not config.storeFullData else device
            self.d_kernel = torch.tensor(self.d_kernel, device=dvc, dtype=torch.float32)

            if len(self.d_kernel < 3):
                self.d_kernel = self.d_kernel.reshape(1, 1, -1)

    def preprocess(self, data, device, config):
        # we set this here again, so on-the-fly loading can have a different device
        self.device = device

    def prepare_batch(self, data, config):
        ret_dict = {DatasetKeyConstants.depth_image_full: data[DatasetKeyConstants.depth_image_full],
                    DatasetKeyConstants.image_sample_indices: data[DatasetKeyConstants.image_sample_indices]}

        return ret_dict

    def batch(self, data, **kwargs) -> (torch.tensor, torch.tensor):
        depths = data[DatasetKeyConstants.depth_image_full]
        sample_indices = data[DatasetKeyConstants.image_sample_indices]

        if self.window_size == 1:
            step = 1.0 / self.n_feat
            features = torch.zeros((torch.numel(sample_indices), self.n_feat),
                                   dtype=torch.float32, device=self.device)
            # use a one hot encoding
            selected_depths = depths.reshape(-1, depths.shape[-1])[sample_indices]
            mask, _ = torch.nonzero(selected_depths < self.ignore_depth_value, as_tuple=True)
            d_disc = torch.clamp_max((selected_depths[mask] / step).type(torch.int64), self.n_feat - 1)

            features[mask, d_disc.T] = 1.
            return features

        else:
            step = 1.0 / self.n_feat
            features = torch.zeros((torch.numel(sample_indices), self.n_feat),
                                   dtype=torch.float32, device=self.device)

            if self.device != "cpu" and self.cuda_batch is not None:
                self.cuda_batch.fill_disc_depth(features, sample_indices,
                                                torch.zeros(1, dtype=torch.int64, device=self.device), depths,
                                                self.window_size, self.h, self.w,
                                                len(sample_indices), 1,
                                                self.center_id, self.n_feat, self.ignore_depth_value, 2)
            else:
                cx = sample_indices % self.w
                cy = sample_indices // self.w
                max_dist = (self.window_size // 2 + 1) * np.sqrt(2.0)

                for i in range(self.window_size):
                    for j in range(self.window_size):
                        weight = (1.0 - np.sqrt((i - self.center_id) ** 2 + (j - self.center_id) ** 2) / max_dist)
                        x = torch.clamp(cx - self.center_id + i, min=0, max=self.w - 1)
                        y = torch.clamp(cy - self.center_id + j, min=0, max=self.h - 1)
                        val = depths[0, y, x].flatten()
                        disc = (val / step).type(torch.int64)
                        mask = torch.nonzero(torch.logical_and(val < self.ignore_depth_value, disc >= 0), as_tuple=True)[0]
                        disc = torch.clamp_max(disc[mask], self.n_feat - 1)
                        features[mask, disc] = torch.max(features[mask, disc],
                                                         torch.tensor([weight], dtype=torch.float32,
                                                                      device=features.device))

            if self.d_window_size > 1:
                features_filtered = torch.nn.functional.conv1d(features[:, None, :], self.d_kernel,
                                                               padding=self.d_window_size // 2)
                features_filtered = torch.clamp(features_filtered, 0., 1.)
                features = features_filtered.reshape(-1, features.shape[1])

            return features

    def get_layers(self, windows):
        features = np.zeros((windows.shape[0], windows.shape[1], self.n_feat + 1), dtype=np.half)

        max_dist = (self.window_size // 2 + 1) * np.sqrt(2.0)
        for i in range(self.window_size):
            for j in range(self.window_size):
                weight = (1.0 - np.sqrt((i - self.center_id) ** 2 + (j - self.center_id) ** 2) / max_dist)
                indices = windows[:, :, i, j][..., np.newaxis]
                new_weights = np.maximum(np.take_along_axis(features, indices, 2), weight)
                np.put_along_axis(features, indices, new_weights, 2)

        return features[..., :self.n_feat]


class RayMarchFromPoses(FeatureSet):
    abbr = "RayMarchFromPoses"

    def __init__(self, config=None, net_idx=-1, device="cpu", **kwargs):
        self.n_ray_samples = config.numRaymarchSamples[net_idx]
        self.z_near = 0.001 if not config.zNear else config.zNear[net_idx]
        self.z_far = 1.0 if not config.zFar else config.zFar[net_idx]
        self.train_with_gt_depth = config.trainWithGTDepth
        self.deterministic_sampling = config.deterministicSampling
        noise_amplitude = 0.0 if not config.rayMarchSamplingNoise else config.rayMarchSamplingNoise[net_idx]
        z_step = (self.z_far - self.z_near) / self.n_ray_samples if not config.rayMarchSamplingStep else \
            config.rayMarchSamplingStep[net_idx]
        self.z_sampler = getattr(import_module("nerf_raymarch_common"), config.rayMarchSampler[net_idx])\
            (self.z_near, self.z_far, self.n_ray_samples, z_step=z_step, noise_amplitude=noise_amplitude, config=config,
             net_idx=net_idx)
        self.depth = None
        self.depth_range = None
        self.depth_transform = None
        self.max_depth = None
        self.n_feat = 6
        self.config = config
        self.net_idx = net_idx
        self.w = 0
        self.h = 0
        self.view = None
        self.view_cell_center = None
        self.device = device

        self.perturb = config.perturb
        self.rayMarchNormalizationCenter = config.rayMarchNormalizationCenter

        if config.rayMarchNormalization:
            self.normalizationFunction = nerf_get_normalization_function(config.rayMarchNormalization[net_idx])
            self.abbr = self.abbr + nerf_get_normalization_function_abbr(config.rayMarchNormalization[net_idx])
        else:
            self.normalizationFunction = nerf_get_normalization_function(None)
            self.abbr = self.abbr + nerf_get_normalization_function_abbr(None)

        if config.posEncArgs[net_idx] == "none":
            self.enc_args = []
            self.n_freq_pos, self.n_freq_dir = -1, -1
        else:
            self.enc_args = [int(x) for x in config.posEncArgs[net_idx].split('-')]
            self.n_freq_pos, self.n_freq_dir = self.enc_args[0], self.enc_args[1]

        enc_type = self.config.posEnc[net_idx]
        self.enc_type = enc_type
        self.pos_enc = FeatureEncoding.get_encoding(enc_type)(config, f"pos{net_idx}")
        self.dir_enc = FeatureEncoding.get_encoding(enc_type)(config, f"dir{net_idx}")

        if enc_type == "nerf":
            self.n_feat = self.n_freq_pos * 3 * 2 + 3 + 3 + self.n_freq_dir * 3 * 2  # 4 for dir encoding according to paper

    def get_string(self):
        return self.abbr + f"[{self.z_sampler.get_name()}]"

    def initialize(self, config, dataset_info, device):
        self.view_cell_center = torch.tensor(dataset_info.view.view_cell_center, device=device)

        self.w = dataset_info.w
        self.h = dataset_info.h

        self.view = dataset_info.view
        self.max_depth = dataset_info.depth_max
        self.depth_range = dataset_info.depth_range if not dataset_info.use_warped_depth_range else dataset_info.depth_range_warped
        self.depth_transform = dataset_info.depth_transform

        self.pos_enc.initialize(n_freqs=self.n_freq_pos)
        self.dir_enc.initialize(n_freqs=self.n_freq_dir)

    def prepare_batch(self, data, config):
        image_indices = data[DatasetKeyConstants.image_sample_indices]

        ret_dict = {DatasetKeyConstants.image_pose: data[DatasetKeyConstants.image_pose][0],
                    DatasetKeyConstants.image_rotation: data[DatasetKeyConstants.image_rotation][0],
                    DatasetKeyConstants.ray_directions_samples: data[DatasetKeyConstants.ray_directions][image_indices]}

        if DatasetKeyConstants.depth_image_full in data:
            depth_image = data[DatasetKeyConstants.depth_image_full][0]
            depth_image = depth_image.reshape(depth_image.shape[0] * depth_image.shape[1], depth_image.shape[2])
            depth_image = depth_image[image_indices]

            ret_dict[DatasetKeyConstants.depth_image_samples] = depth_image

        return ret_dict

    def batch(self, data, **kwargs) -> (torch.tensor, torch.tensor):
        poses = data[DatasetKeyConstants.image_pose]
        rotations = data[DatasetKeyConstants.image_rotation]
        directions = data[DatasetKeyConstants.ray_directions_samples]

        depth_image = None
        if DatasetKeyConstants.depth_image_samples in data:
            depth_image = data[DatasetKeyConstants.depth_image_samples]

        n_images = poses.shape[0]
        n_samples = directions.shape[1]

        prev_outs = kwargs.get('prev_outs', None)
        is_inference = kwargs.get('is_inference', False)

        depth = None
        # If we have prev_outs, use the depth from that ray
        # Otherwise (default) use the GT depth
        if prev_outs is not None and len(prev_outs) > 0 and (not self.train_with_gt_depth or is_inference is True):
            depth = prev_outs[-1][FeatureSetKeyConstants.network_output]
        else:
            if (not is_inference or len(prev_outs) == 0) and depth_image is not None:
                depth = depth_image

        z_vals = self.z_sampler.generate(n_images * n_samples, poses.device, depth=depth, depth_range=self.depth_range,
                                         depth_transform=self.depth_transform,
                                         det=self.deterministic_sampling or is_inference)

        if self.perturb and is_inference is False:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=z_vals.device)
            z_vals = lower + (upper - lower) * t_rand

        ray_origins = None
        ray_directions = None
        if prev_outs is not None and len(prev_outs) > 0:
            p_out = prev_outs[-1]
            if FeatureSetKeyConstants.input_feature_ray_origins in p_out:
                ray_origins = p_out[FeatureSetKeyConstants.input_feature_ray_origins]
            if FeatureSetKeyConstants.input_feature_ray_directions in p_out:
                ray_directions = p_out[FeatureSetKeyConstants.input_feature_ray_directions]

        if ray_directions is None:
            ray_directions = nerf_get_ray_dirs(rotations, directions)

        if ray_origins is None:
            # Now that we have the directions for the chosen images, we need the
            # corresponding poses to then generate all samples
            ray_origins = tile(poses, dim=0, n_tile=n_samples).reshape(n_images * n_samples, -1)

        ray_sample_positions = (ray_origins[..., None, :] + ray_directions[..., None, :] * z_vals[..., :, None])

        if len(self.rayMarchNormalizationCenter) == 3:
            ray_sample_positions = self.normalizationFunction(ray_sample_positions,
                                                              torch.tensor(self.rayMarchNormalizationCenter,
                                                                           device=ray_sample_positions.device),
                                                              self.max_depth)
        else:
            ray_sample_positions = self.normalizationFunction(ray_sample_positions, self.view_cell_center,
                                                              self.max_depth)

        # Reshape to positions only and do positional encoding
        inputs = ray_sample_positions
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

        embedded = self.pos_enc.encode(inputs_flat)

        input_dirs = ray_directions[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = self.dir_enc.encode(input_dirs_flat)

        embedded = torch.cat([embedded, embedded_dirs], -1)

        embedded = torch.reshape(embedded, [-1, self.n_ray_samples, embedded.shape[-1]])

        ret_dict = {FeatureSetKeyConstants.input_feature_batch: embedded,
                    FeatureSetKeyConstants.nerf_input_feature_z_vals: z_vals,
                    FeatureSetKeyConstants.nerf_input_feature_ray_directions: ray_directions,
                    FeatureSetKeyConstants.nerf_input_feature_ray_origins: ray_origins}

        if not is_inference and depth_image is not None:
            ret_dict[FeatureSetKeyConstants.input_depth_groundtruth] = depth_image
            ret_dict[FeatureSetKeyConstants.input_depth_groundtruth_world] = self.depth_transform.to_world(
                depth_image, self.depth_range)

        ret_dict[FeatureSetKeyConstants.input_depth_range] = torch.tensor(self.depth_range)
        ret_dict[FeatureSetKeyConstants.input_depth] = depth

        return ret_dict

    def postprocess(self, inference_dict, idx):
        network_out = inference_dict[FeatureSetKeyConstants.network_output]

        ray_directions = inference_dict[FeatureSetKeyConstants.nerf_input_feature_ray_directions]
        z_vals = inference_dict[FeatureSetKeyConstants.nerf_input_feature_z_vals]

        rgb_map, disp_map, acc_map, weights, depth_map = nerf_raw2outputs(
            network_out.reshape(ray_directions.shape[0], z_vals.shape[1], -1), z_vals, ray_directions)

        inference_dict[FeatureSetKeyConstants.postprocessed_network_output] = rgb_map
        inference_dict[FeatureSetKeyConstants.nerf_weights_output] = weights

        inference_dict[FeatureSetKeyConstants.nerf_estimated_depth] = torch.reshape(
            self.depth_transform.from_world(depth_map, self.depth_range), (-1, 1))


class RayMarchFromCoarse(FeatureSet):
    abbr = "RayMarchFromCoarse"

    def __init__(self, config=None, net_idx=-1, device="cpu", **kwargs):
        self.n_ray_samples = config.numRaymarchSamples[net_idx]
        self.z_near = config.zNear[net_idx]
        self.z_far = config.zFar[net_idx]
        self.depth_range = [0, 1]
        self.max_depth = 1
        self.n_feat = 6
        self.config = config
        self.net_idx = net_idx
        self.w = 0
        self.h = 0
        self.view_cell_center = None
        self.depth_transform = None
        self.view = None
        self.device = device
        self.perturb = config.perturb

        if config.rayMarchNormalization:
            self.normalizationFunction = nerf_get_normalization_function(config.rayMarchNormalization[net_idx])
            self.abbr = self.abbr + nerf_get_normalization_function_abbr(config.rayMarchNormalization[net_idx])
        else:
            self.normalizationFunction = nerf_get_normalization_function(None)
            self.abbr = self.abbr + nerf_get_normalization_function_abbr(None)

        if config.posEncArgs[net_idx] == "none":
            self.enc_args = []
            self.n_freq_pos, self.n_freq_dir = -1, -1
        else:
            self.enc_args = [int(x) for x in config.posEncArgs[net_idx].split('-')]
            self.n_freq_pos, self.n_freq_dir = self.enc_args[0], self.enc_args[1]

        enc_type = self.config.posEnc[net_idx]
        self.enc_type = enc_type
        self.pos_enc = FeatureEncoding.get_encoding(enc_type)(config, f"pos{net_idx}")
        self.dir_enc = FeatureEncoding.get_encoding(enc_type)(config, f"dir{net_idx}")

        if enc_type == "nerf":
            self.n_feat = self.n_freq_pos * 3 * 2 + 3 + 3 + self.n_freq_dir * 3 * 2  # 4 for dir encoding according to paper

    def initialize(self, config, dataset_info, device):
        self.pos_enc.initialize(n_freqs=self.n_freq_pos)
        self.dir_enc.initialize(n_freqs=self.n_freq_dir)

        self.w = dataset_info.w
        self.h = dataset_info.h

        self.view = dataset_info.view
        self.depth_range = dataset_info.depth_range if not dataset_info.use_warped_depth_range else dataset_info.depth_range_warped
        self.depth_transform = dataset_info.depth_transform
        self.max_depth = dataset_info.depth_max
        self.view_cell_center = torch.tensor(dataset_info.view.view_cell_center, device=device)

    def get_string(self):
        return self.abbr + f"[{self.z_near}_{self.z_far}_{self.n_ray_samples}]"

    def prepare_batch(self, data, config):
        return {}

    def batch(self, data, **kwargs) -> (torch.tensor, torch.tensor):
        prev_outs = kwargs.get('prev_outs', None)

        # Get previous output inference dict
        if prev_outs is None:
            raise Exception(f"Error: feature {self.abbr} requires prev_outs to work!")

        # Mostly taken from nerf-pytorch https://github.com/yenchenlin/nerf-pytorch
        p_out = prev_outs[-1]
        previous_z_vals = p_out[FeatureSetKeyConstants.nerf_input_feature_z_vals]
        weights = p_out[FeatureSetKeyConstants.nerf_weights_output]
        ray_origins = p_out[FeatureSetKeyConstants.nerf_input_feature_ray_origins]
        ray_directions = p_out[FeatureSetKeyConstants.nerf_input_feature_ray_directions]

        z_vals_mid = .5 * (previous_z_vals[..., 1:] + previous_z_vals[..., :-1])
        z_samples = nerf_sample_pdf(z_vals_mid, weights[..., 1:-1], self.n_ray_samples, det=(not self.perturb))
        z_samples = z_samples.detach()
        z_vals, _ = torch.sort(torch.cat([previous_z_vals, z_samples], -1), -1)
        ray_sample_positions = ray_origins[..., None, :] + ray_directions[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + self.n_ray_samples, 3]
        ray_sample_positions = self.normalizationFunction(ray_sample_positions, self.view_cell_center, self.max_depth)

        # Reshape to positions only and do positional encoding
        inputs = ray_sample_positions
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

        embedded = self.pos_enc.encode(inputs_flat)

        input_dirs = ray_directions[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = self.dir_enc.encode(input_dirs_flat)

        embedded = torch.cat([embedded, embedded_dirs], -1)

        ret_dict = {FeatureSetKeyConstants.input_feature_batch: embedded,
                    FeatureSetKeyConstants.nerf_input_feature_z_vals: z_vals,
                    FeatureSetKeyConstants.nerf_input_feature_ray_directions: ray_directions,
                    FeatureSetKeyConstants.nerf_input_feature_ray_origins: ray_origins,
                    FeatureSetKeyConstants.input_depth_range: torch.tensor(self.depth_range)}

        return ret_dict

    def postprocess(self, inference_dict, idx):
        network_out = inference_dict[FeatureSetKeyConstants.network_output]

        ray_directions = inference_dict[FeatureSetKeyConstants.nerf_input_feature_ray_directions]
        z_vals = inference_dict[FeatureSetKeyConstants.nerf_input_feature_z_vals]

        rgb_map, disp_map, acc_map, weights, depth_map = nerf_raw2outputs(
            network_out.reshape(ray_directions.shape[0], z_vals.shape[1], -1), z_vals, ray_directions)

        inference_dict[FeatureSetKeyConstants.postprocessed_network_output] = rgb_map
        inference_dict[FeatureSetKeyConstants.nerf_weights_output] = weights

        inference_dict[FeatureSetKeyConstants.nerf_estimated_depth] = torch.reshape(
            self.depth_transform.from_world(depth_map, self.depth_range), (-1, 1))


class SpherePosDir(FeatureSet):
    def __init__(self, config=None, net_idx=-1, device="cpu", **kwargs):
        self.n_feat = 3 + 3  # 3 cam pos + 3 dir vec
        self.config = config
        self.net_idx = net_idx
        self.w = 0
        self.h = 0
        self.view = None
        self.depth = None
        self.depth_range = None
        self.depth_transform = None
        self.device = device

        if config.posEncArgs[net_idx] == "none":
            self.enc_args = []
            self.n_freq_pos, self.n_freq_dir = -1, -1
        else:
            self.enc_args = [int(x) for x in config.posEncArgs[net_idx].split('-')]
            self.n_freq_pos, self.n_freq_dir = self.enc_args[0], self.enc_args[1]

        enc_type = self.config.posEnc[net_idx]
        self.enc_type = enc_type
        self.pos_enc = FeatureEncoding.get_encoding(enc_type)(config, f"pos{net_idx}")
        self.dir_enc = FeatureEncoding.get_encoding(enc_type)(config, f"dir{net_idx}")

        if enc_type == "nerf":
            self.n_feat = self.n_freq_pos * 3 * 2 + 3 + 3 + self.n_freq_dir * 3 * 2  # 4 for dir encoding according to paper

        self.abbr = "SpPoDi"
        self.additionalSamples = 0
        self.view_cell_center = None
        self.view_cell_radius = None
        self.view_cell_center_gpu = None
        self.view_cell_radius_gpu = None
        self.depth_range_warped = None

        if config.raySampleInput:
            self.additionalSamples = config.raySampleInput[net_idx]
        if self.additionalSamples != 0:
            self.abbr = f"SpPoDir[{self.additionalSamples}]"

            if self.enc_type == "nerf":
                self.n_feat = (self.additionalSamples * 3 + 3) * (
                            self.n_freq_pos * 2 + 1) + 3 + self.n_freq_dir * 3 * 2  # 4 for dir encoding according to paper
            else:
                self.n_feat = 3 + 3 + self.additionalSamples * 3

    def initialize(self, config, dataset_info, device="cpu"):
        self.pos_enc.initialize(n_freqs=self.n_freq_pos)
        self.dir_enc.initialize(n_freqs=self.n_freq_dir)

        self.w = dataset_info.w
        self.h = dataset_info.h

        self.view = dataset_info.view
        self.depth_range = dataset_info.depth_range
        self.depth_range_warped = dataset_info.depth_range_warped
        self.depth_transform = dataset_info.depth_transform

        self.view_cell_center = torch.tensor(np.array(dataset_info.view.view_cell_center), device="cpu",
                                             dtype=torch.float32)
        self.view_cell_radius = torch.tensor(np.array([max(dataset_info.view.view_cell_size[0],
                                                           max(dataset_info.view.view_cell_size[1],
                                                               dataset_info.view.view_cell_size[2]))]),
                                             device="cpu", dtype=torch.float32)

        # we need cpu and gpu versions of these tensors, as we may need them in the preprocess function (cpu)
        # and the batch function (gpu)
        if device != "cpu":
            self.view_cell_center_gpu = self.view_cell_center.clone().detach().to(device)
            self.view_cell_radius_gpu = self.view_cell_radius.clone().detach().to(device)

    @classmethod
    def compute_ray_offset(cls, nds, poses, view_cell_center, view_cell_radius):
        # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        # d - distance along line from starting point
        # u - direction of line (a unit vector)
        # o - origin of the line
        # c - sphere center
        # r - sphere radius

        # d = -(u*(o-c)) pm sqrt(delta)
        # delta = (u*(o-c))^2 - (||o-c||^2 - r^2)

        oMc = poses - view_cell_center  # (o-c)
        oMcE = tile(oMc, dim=0, n_tile=nds.shape[1])

        uDot_oMc = torch.sum(oMcE * nds.flatten().reshape(-1, 3), dim=1)  # u*(o-c)

        delta = uDot_oMc ** 2 - tile((torch.sum(oMc ** 2, dim=-1) - view_cell_radius ** 2), dim=0,
                                     n_tile=nds.shape[1])  # (u*(o-c))^2 - (||o-c||^2 - r^2)
        sqrt_delta = torch.sqrt(torch.clamp_min(delta, 0))  # lets just make sure we don't get NaNs due to precision

        d = -uDot_oMc + sqrt_delta

        return d.reshape(-1, nds.shape[1], 1)

    def preprocess(self, data, device, config):
        directions = data[DatasetKeyConstants.ray_directions]
        rotations = data[DatasetKeyConstants.image_rotation]
        poses = data[DatasetKeyConstants.image_pose]
        depths = data[DatasetKeyConstants.depth_image_full]

        view_cell_c = self.view_cell_center
        view_cell_r = self.view_cell_radius

        if device != "cpu":
            view_cell_c = self.view_cell_center_gpu
            view_cell_r = self.view_cell_radius_gpu

        for idx in range(depths.shape[0]):
            rotation = rotations[idx]
            pose = poses[idx]
            depth = depths[idx]

            nds = torch.matmul(directions, torch.transpose(rotation, 0, 1))
            distance = self.compute_ray_offset(nds[None], pose[None, :], view_cell_c, view_cell_r)
            mask = depth == 1.

            depth = self.depth_transform.to_world(depth, self.depth_range)
            depth = depth - torch.reshape(distance, (depth.shape[0], depth.shape[1], 1))
            depth[mask] = self.depth_range[1]
            depths[idx] = depth

        mask = depths == self.depth_range[1]
        depths = self.depth_transform.from_world(depths, self.depth_range_warped)
        depths[mask] = 1.
        data[DatasetKeyConstants.depth_image_full] = depths

    def prepare_batch(self, data, config):
        image_indices = data[DatasetKeyConstants.image_sample_indices]

        ret_dict = {DatasetKeyConstants.image_pose: data[DatasetKeyConstants.image_pose][0],
                    DatasetKeyConstants.image_rotation: data[DatasetKeyConstants.image_rotation][0],
                    DatasetKeyConstants.ray_directions_samples: data[DatasetKeyConstants.ray_directions][image_indices]}

        if DatasetKeyConstants.depth_image_full in data:
            depth_image = data[DatasetKeyConstants.depth_image_full][0]
            depth_image = depth_image.reshape(depth_image.shape[0] * depth_image.shape[1], depth_image.shape[2])
            depth_image = depth_image[image_indices]

            ret_dict[DatasetKeyConstants.depth_image_samples] = depth_image

        return ret_dict

    def batch(self, data, **kwargs) -> (torch.tensor, torch.tensor):
        poses = data[DatasetKeyConstants.image_pose]
        rotations = data[DatasetKeyConstants.image_rotation]
        directions = data[DatasetKeyConstants.ray_directions_samples]

        n_images = poses.shape[0]
        n_samples = directions.shape[1]

        x_select = torch.zeros((n_images, n_samples, self.n_feat), device=self.config.device)

        is_inference = kwargs.get('is_inference', False)

        # multiply them with the camera transformation matrix
        nds = torch.bmm(rotations, torch.transpose(directions, 1, 2))
        nds = torch.transpose(nds, 1, 2)

        distance = self.compute_ray_offset(nds, poses, self.view_cell_center_gpu, self.view_cell_radius_gpu)

        pt = tile(poses, dim=0, n_tile=n_samples)
        proj_points = pt + (nds * tile(distance, dim=2, n_tile=3)).reshape(nds.shape[0] * nds.shape[1], -1)

        encnds = self.dir_enc.encode(nds).reshape(n_images, n_samples, -1)
        enc_proj_points = self.pos_enc.encode(proj_points)
        x_select[:, :, :encnds.shape[-1]] = encnds
        x_select = x_select.reshape(-1, x_select.shape[-1])
        encoded_x = enc_proj_points
        x_select[:, encnds.shape[-1]:(encnds.shape[-1] + encoded_x.shape[1])] = encoded_x

        if self.additionalSamples != 0:
            step = 1. / self.additionalSamples
            z_vals = self.depth_transform.to_world(torch.linspace(step / 2, 1. - step / 2,
                                                                  self.additionalSamples,
                                                                  device=nds.device),
                                                   self.depth_range_warped)

            add_samples = (proj_points[..., None, :] + nds.reshape(-1, 3)[..., None, :] * z_vals[None, :, None])

            enc_add_samples = self.pos_enc.encode(add_samples / self.depth_range_warped[1])
            enc_add_samples[:, :, :3] *= self.depth_range_warped[1]
            enc_add_samples = enc_add_samples.reshape(add_samples.shape[0], -1)
            x_select[:, -enc_add_samples.shape[1]:] = enc_add_samples

        ret_dict = {FeatureSetKeyConstants.input_feature_batch: x_select}

        if not is_inference and DatasetKeyConstants.depth_image_samples in data:
            ret_dict[FeatureSetKeyConstants.input_depth_groundtruth] = data[DatasetKeyConstants.depth_image_samples]
            ret_dict[FeatureSetKeyConstants.input_depth_groundtruth_world] = self.depth_transform.to_world(
                data[DatasetKeyConstants.depth_image_samples], self.depth_range_warped)

        ret_dict[FeatureSetKeyConstants.input_depth_range] = torch.tensor(self.depth_range_warped)

        ret_dict[FeatureSetKeyConstants.input_feature_ray_origins] = proj_points
        ret_dict[FeatureSetKeyConstants.input_feature_ray_directions] = nds.reshape(-1, 3)

        return ret_dict
