# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import math

import util.depth_transformations as depth_transforms

from importlib import import_module


# Mostly adapted from nerf-pytorch https://github.com/yenchenlin/nerf-pytorch (licensed under MIT, see LICENSE_third_party.md in the root of this repository)
def nerf_raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    # This is a helper function ported over from the nerf-pytorch project
    raw2alpha = lambda raw, dists, act_fn=torch.nn.functional.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, (torch.ones(1, device=raw.device) * 1e10).expand(dists[..., :1].shape)],
                      -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=raw.device), 1. - alpha + 1e-10], -1), -1)[:,
                      :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map, device=raw.device), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def nerf_get_ray_dirs(rotations, directions) -> torch.tensor:
    # multiply them with the camera transformation matrix
    ray_directions = torch.bmm(rotations, torch.transpose(directions, 1, 2))
    ray_directions = torch.transpose(ray_directions, 1, 2).reshape(directions.shape[0] * directions.shape[1], -1)

    return ray_directions


def nerf_get_z_vals(idx, z_near, z_far, poses, n_ray_samples, sampler_type='LinearlySpacedZNearZFar', **kwargs) -> torch.tensor:
    return getattr(import_module("nerf_raymarch_common"), sampler_type).generate(idx, z_near, z_far, idx.n_samples, n_ray_samples, idx.n_images, poses.device, **kwargs)


# Hierarchical sampling (section 5.2)
def nerf_sample_pdf(bins, weights, n_samples, det=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=n_samples, device=weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf.detach(), u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def normalization_none(x_in_world, view_cell_center, max_depth):
    return x_in_world


def normalization_center(x_in_world, view_cell_center, max_depth):
    return x_in_world - view_cell_center


def normalization_max_depth(x_in_world, view_cell_center, max_depth):
    return x_in_world / max_depth


def normalization_max_depth_centered(x_in_world, view_cell_center, max_depth):
    return (x_in_world - view_cell_center) / max_depth


def normalization_log_centered(x_in_world, view_cell_center, max_depth):
    localized = x_in_world - view_cell_center
    local = torch.linalg.norm(localized, dim=-1)
    log_transformed = depth_transforms.LogTransform.from_world(local, [0, max_depth])
    p = localized * (log_transformed / local)[..., None]
    return p


def normalization_inverse_dist_centered(x_in_world, view_cell_center, max_depth):
    localized = x_in_world - view_cell_center
    local = torch.linalg.norm(localized, dim=-1)
    p = localized * (1. - 1. / (1. + local))[..., None]
    return p


def normalization_inverse_sqrt_dist_centered(x_in_world, view_cell_center, max_depth):
    localized = x_in_world - view_cell_center
    local = torch.sqrt(torch.linalg.norm(localized, dim=-1))
    res = localized / (math.sqrt(max_depth) * local[..., None])
    return res


def nerf_get_normalization_function(name):
    switcher = {
        None: normalization_max_depth,
        "None": normalization_none,
        "Centered": normalization_center,
        "MaxDepth": normalization_max_depth,
        "MaxDepthCentered": normalization_max_depth_centered,
        "LogCentered": normalization_log_centered,
        "InverseDistCentered": normalization_inverse_dist_centered,
        "InverseSqrtDistCentered": normalization_inverse_sqrt_dist_centered
    }
    return switcher.get(name)


def nerf_get_normalization_function_abbr(name):
    switcher = {
        None: "",
        "None": "_nN",
        "Centered": "_nC",
        "MaxDepth": "",
        "MaxDepthCentered": "_nMdC",
        "LogCentered": "_nL",
        "InverseDistCentered": "_nD",
        "InverseSqrtDistCentered": "_nSD"
    }
    return switcher.get(name)


class LinearlySpacedZNearZFar:
    """
    This just samples linearly between z_near and z_far without anything else.
    """
    def __init__(self, z_near, z_far, num_ray_samples, noise_amplitude, **kwargs):
        self.z_near = z_near
        self.z_far = z_far
        self.num_ray_samples = num_ray_samples
        self.noise_amplitude = noise_amplitude
        self.print_name = f"{self.z_near}_{self.z_far}_{self.num_ray_samples}_{self.__class__.__name__}"

    def generate(self, idx, device, **kwargs):
        depth_range = kwargs.get('depth_range', None)
        depth_transform = kwargs.get('depth_transform', None)
        t_vals = torch.linspace(0., 1., steps=int(self.num_ray_samples), device=device)
        near_vec = torch.ones((idx, 1), device=device) * self.z_near  # [n_images * n_samples, 1]
        far_vec = torch.ones((idx, 1), device=device) * self.z_far  # [n_images * n_samples, 1]
        z_vals = near_vec * (1. - t_vals) + far_vec * t_vals

        return depth_transform.to_world(z_vals, depth_range)

    def get_name(self):
        return self.print_name


class LinearlySpacedFromDepth:
    """
    This just samples linearly between new z_near (computed from depth and original z_near/z_far spacing) and new z_far
    """
    def __init__(self, z_near, z_far, num_ray_samples, z_step, noise_amplitude, **kwargs):
        self.z_near = z_near
        self.z_far = z_far
        self.num_ray_samples = num_ray_samples
        self.z_step = z_step
        self.noise_amplitude = noise_amplitude
        self.print_name = f"{self.z_near}_{self.z_far}_{self.num_ray_samples}_{self.__class__.__name__}_{self.z_step}_{self.noise_amplitude}"

    def generate(self, idx, device, **kwargs):
        depth = kwargs.get('depth', None)
        depth_range = kwargs.get('depth_range', None)
        depth_transform = kwargs.get('depth_transform', None)
        z_step = self.z_step
        noise = self.noise_amplitude

        # Add noise from -z_step/2 to +z_step/2, scaled by noise factor
        noise_add = noise * (-z_step / 2 + z_step * torch.rand_like(depth))

        depth = depth.detach()
        s_depth = depth + noise_add

        z_near = s_depth - z_step * self.num_ray_samples / 2

        z_vals = (z_near[..., None] + torch.linspace(0, z_step * self.num_ray_samples, int(self.num_ray_samples),
                                                     device=device, dtype=torch.float32)).reshape(
            idx, self.num_ray_samples)  # [n_images * n_samples, num_ray_samples]

        return depth_transform.to_world(z_vals, depth_range)

    def get_name(self):
        return self.print_name


class LinearlySpacedFromMultiDepth:
    """
    This just samples linearly around multiple reference points
    """
    def __init__(self, z_near, z_far, num_ray_samples, z_step, noise_amplitude, config=None, net_idx=-1, **kwargs):
        self.z_near = z_near
        self.z_far = z_far
        self.num_ray_samples = num_ray_samples
        self.z_step = z_step
        self.noise_amplitude = noise_amplitude
        self.net_idx = net_idx
        self.background_value = config.multiDepthIgnoreValue[net_idx]
        self.print_name = f"{self.z_near}_{self.z_far}_{self.num_ray_samples}_LSfMD_{self.z_step}_{self.noise_amplitude}"

    def generate(self, idx, device, **kwargs):
        depth = kwargs.get('depth', None)
        depth_range = kwargs.get('depth_range', None)
        depth_transform = kwargs.get('depth_transform', None)
        z_step = self.z_step
        noise = self.noise_amplitude

        sorted_depth, ids = torch.sort(depth)
        sorted_depth = torch.clamp(sorted_depth, min=0., max=1.)

        # Add noise from -z_step/2 to +z_step/2, scaled by noise factor
        noise_add = noise * (-z_step / 2 + z_step * torch.rand_like(sorted_depth))

        sorted_depth = sorted_depth + noise_add

        starting_points = depth.shape[-1]
        samples_per_point = (self.num_ray_samples+starting_points-1) // starting_points

        z_nears = sorted_depth - z_step * samples_per_point / 2

        # ensure samples are z_step * (samples_per_point + 1) apart
        mind_dist = z_step * (samples_per_point + 1)
        for i in range(starting_points-1):
            dist = z_nears[:, starting_points - i - 1] - z_nears[:, starting_points - i - 2]
            off = torch.clamp(dist - mind_dist, max=0)
            z_nears[:, starting_points - i - 2] += off

        z_nears_base = torch.repeat_interleave(z_nears, samples_per_point, dim=1)

        steps = torch.linspace(0, z_step * samples_per_point, samples_per_point,
                               device=device, dtype=torch.float32)
        steps_repeated = steps.repeat(z_nears_base.shape[0], starting_points)

        z_vals = (z_nears_base + steps_repeated).reshape(
            idx, starting_points * samples_per_point)  # [n_images * n_samples, starting_points * samples_per_point]

        return depth_transform.to_world(z_vals, depth_range)

    def get_name(self):
        return self.print_name


class FromClassifiedDepth:
    """
    This just samples linearly around multiple reference points
    """
    def __init__(self, z_near, z_far, num_ray_samples, z_step, noise_amplitude, config=None, net_idx=-1, **kwargs):
        self.z_near = z_near
        self.z_far = z_far
        self.num_ray_samples = num_ray_samples
        self.z_step = z_step
        self.noise_amplitude = noise_amplitude
        self.net_idx = net_idx
        self.background_value = config.multiDepthIgnoreValue[net_idx]

        self.disc = 128
        if config.multiDepthFeatures:
            self.disc = config.multiDepthFeatures[net_idx]
        self.print_name = f"{self.num_ray_samples}_LSfCD_{self.disc}_{self.noise_amplitude}"
        self.transform = None

        if self.net_idx > 0:
            if config.losses[self.net_idx-1] == "BCEWithLogitsLoss":
                self.transform = torch.sigmoid
            elif config.losses[self.net_idx-1] == "CrossEntropyLoss":
                self.transform = self.softmax
            elif config.losses[self.net_idx-1] == "CrossEntropyLossWeighted":
                self.transform = self.softmaxselect

    def softmax(self, depth):
        return torch.nn.functional.softmax(depth, dim=-1)

    def softmaxselect(self, depth):
        return torch.nn.functional.softmax(depth[..., :self.disc], dim=-1)

    def generate(self, idx, device, **kwargs):
        depth = kwargs.get('depth', None)
        depth_range = kwargs.get('depth_range', None)
        depth_transform = kwargs.get('depth_transform', None)
        det = kwargs.get('deterministic_sampling', True)
        depth = depth.detach()

        if self.transform:
            depth = self.transform(depth)

        disc_steps = depth.shape[-1]

        mids_single = torch.linspace(0., 1., disc_steps+1, device=device, dtype=torch.float32)
        mids_all = mids_single.repeat(depth.shape[0], 1)

        z_samples = nerf_sample_pdf(mids_all, depth, self.num_ray_samples+2, det=det)
        z_samples = z_samples[:,1:-1]
        z_samples = z_samples.detach()
        return depth_transform.to_world(z_samples, depth_range)

    def get_name(self):
        return self.print_name
