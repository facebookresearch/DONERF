# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import torch
import configargparse

import numpy as np
import depth_transformations as depth_transforms

from tqdm import trange
from features import SpherePosDir
from util.raygeneration import generate_ray_directions


def load_depth_image(filename, h, w, flip_depth):
    np_file = np.load(filename)
    depth_image = np_file["depth"] if "depth" in np_file.files else np_file[np_file.files[0]]
    depth_image = depth_image.astype(np.float32)
    depth_image = depth_image.reshape(h, w)

    if flip_depth:
        depth_image = np.flip(depth_image, 0)

    return depth_image


def get_min_max_values(depth_image, max_depth_locations, depth_range, device, frame,
                       directions, view_cell_center, view_cell_radius, depth_transform):
    # now the transform is in linear, go back to world and then encoded 0-1
    depth_image = depth_transform.from_world(depth_transforms.LinearTransform.to_world(depth_image, depth_range),
                                             depth_range)

    # set out elements
    depth_image[max_depth_locations] = 1.

    depth_image = torch.from_numpy(depth_image).to(device)

    transform = np.array(frame["transform_matrix"]).astype(np.float32)
    pose = transform[:3, 3:].reshape(-1, 3)
    pose = torch.tensor(pose, device=device, dtype=torch.float32)
    rotation = torch.tensor(transform[:3, :3], device=device, dtype=torch.float32)

    nds = torch.matmul(directions, torch.transpose(rotation, 0, 1))
    distance = SpherePosDir.compute_ray_offset(nds[None], pose, view_cell_center, view_cell_radius)
    mask = depth_image == 1.
    depth_image = depth_transform.to_world(depth_image, depth_range)
    distance = torch.reshape(distance, (depth_image.shape[0], depth_image.shape[1]))
    depth_image = depth_image - distance

    min_v = torch.min(depth_image)
    depth_image[mask] = 0
    max_v = torch.max(depth_image)

    return min_v, max_v


def main():
    parser = configargparse.ArgParser()
    parser.add_argument("-data", "--dataset", required=True, type=str,
                        help="the path to the directory where the depth information will be stored")
    parser.add_argument("-d", "--device", type=int, help="the device id to be used (-1 means cpu)")

    arguments = parser.parse_args()
    path = arguments.dataset
    device_id = arguments.device

    splits = ["train", "val", "test"]

    max_depth = float('-inf')
    min_z = float('inf')
    max_z = float('-inf')

    dataset_info_path = os.path.join(path, "dataset_info.json")

    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)

    w, h = dataset_info["resolution"][0], dataset_info["resolution"][1]
    flip_depth = dataset_info["flip_depth"]

    depth_distance_adjustment = dataset_info["depth_distance_adjustment"]

    # get max depth first, because it influences depth range later on
    for s in splits:
        with open(os.path.join(path, f"transforms_{s}.json"), "r") as f:
            transforms_data = json.load(f)

        tqdm_range = trange(len(transforms_data["frames"]), desc=f"Searching for max depth value ({s:5s})", leave=True)

        for frame_idx in tqdm_range:
            frame = transforms_data["frames"][frame_idx]

            depth_file = os.path.join(path, frame['file_path'] + '_depth.npz')
            depth_image = load_depth_image(depth_file, h, w, flip_depth)
            max_depth = max(np.max(depth_image), max_depth)

    camera_scale = 1.
    if "camera_scale" in dataset_info:
        camera_scale = float(dataset_info['camera_scale'])

    fov = float(dataset_info['camera_angle_x'])
    focal = float(.5 * w / np.tan(.5 * fov))

    ray_dirs = generate_ray_directions(w, h, fov, focal)
    base_ray_z = np.abs(ray_dirs[:, :, 2]).astype(np.float32)

    for s in splits:
        with open(os.path.join(path, f"transforms_{s}.json"), "r") as f:
            transforms_data = json.load(f)

        tqdm_range = trange(len(transforms_data["frames"]), desc=f"Determining depth range ({s:5s})", leave=True)

        for frame_idx in tqdm_range:
            frame = transforms_data["frames"][frame_idx]

            depth_file = os.path.join(path, frame['file_path'] + "_depth.npz")
            depth_image = load_depth_image(depth_file, h, w, flip_depth)

            max_depth_locations = depth_image == max_depth

            if depth_distance_adjustment:
                depth_image = depth_image / base_ray_z[:, :]

            depth_image[max_depth_locations] = -10 * max_depth
            depth_scale_max = 1.05 * np.max(depth_image)
            depth_image[max_depth_locations] = 10 * max_depth
            depth_scale_min = 0.95 * np.min(depth_image)

            min_z = min(depth_scale_min, min_z)
            max_z = max(depth_scale_max, max_z)

    depth_range = [min_z / camera_scale, max_z / camera_scale]

    dataset_info["depth_ignore"] = float(max_depth)
    dataset_info["depth_range"] = depth_range

    if device_id >= 0:
        device = f"cuda:{device_id}"
    else:
        device = "cpu"

    directions = torch.tensor(ray_dirs.flatten().reshape(-1, 3), device=device, dtype=torch.float32)

    view_cell_center = dataset_info["view_cell_center"]
    view_cell_size = dataset_info["view_cell_size"]

    min_v_log = torch.tensor([depth_range[1]], dtype=torch.float32, device=device)
    max_v_log = torch.tensor([depth_range[0]], dtype=torch.float32, device=device)

    min_v_lin = torch.tensor([depth_range[1]], dtype=torch.float32, device=device)
    max_v_lin = torch.tensor([depth_range[0]], dtype=torch.float32, device=device)

    view_cell_center = torch.tensor(np.array(view_cell_center), device=device, dtype=torch.float32)
    view_cell_radius = torch.tensor(np.array([max(view_cell_size[0], max(view_cell_size[1], view_cell_size[2]))]),
                                    device=device, dtype=torch.float32)

    # adjust all depth images
    for s in splits:
        with open(os.path.join(path, f"transforms_{s}.json"), "r") as f:
            transforms_data = json.load(f)

        tqdm_range = trange(len(transforms_data["frames"]), desc=f"Determining warped depth range ({s:5s})", leave=True)

        for frame_idx in tqdm_range:
            frame = transforms_data["frames"][frame_idx]

            depth_file = os.path.join(path, frame['file_path'] + "_depth.npz")
            depth_image = load_depth_image(depth_file, h, w, flip_depth)

            # take care of background, use 1.05 largest other depth value
            max_depth_locations = depth_image == max_depth

            if depth_distance_adjustment:
                depth_image = depth_image / base_ray_z[:, :]

            # we use min_z and max_z, because depth_range is already using camera_scale
            depth_image = (depth_image - min_z) / (max_z - min_z)

            # we calculate warped ranges for both linear and logarithmic transforms
            # while the difference is not very large, it may still cause issues
            min_log, max_log = get_min_max_values(depth_image, max_depth_locations, depth_range, device, frame,
                                                  directions, view_cell_center, view_cell_radius,
                                                  depth_transforms.LogTransform)
            min_v_log = torch.minimum(min_log, min_v_log)
            max_v_log = torch.maximum(max_log, max_v_log)

            min_lin, max_lin = get_min_max_values(depth_image, max_depth_locations, depth_range, device, frame,
                                                  directions, view_cell_center, view_cell_radius,
                                                  depth_transforms.LinearTransform)
            min_v_lin = torch.minimum(min_lin, min_v_lin)
            max_v_lin = torch.maximum(max_lin, max_v_lin)

    new_depth_range_log = [depth_range[0], depth_range[1]]

    min_v_log = min_v_log.item()
    max_v_log = max_v_log.item()

    if min_v_log < depth_range[0]:
        new_depth_range_log[0] = 0.95 * min_v_log
    if max_v_log < depth_range[1]:
        new_depth_range_log[1] = 1.05 * max_v_log

    new_depth_range_lin = [depth_range[0], depth_range[1]]

    min_v_lin = min_v_lin.item()
    max_v_lin = max_v_lin.item()

    if min_v_lin < depth_range[0]:
        new_depth_range_lin[0] = 0.95 * min_v_lin
    if max_v_lin < depth_range[1]:
        new_depth_range_lin[1] = 1.05 * max_v_lin

    dataset_info["depth_range_warped_log"] = new_depth_range_log
    dataset_info["depth_range_warped_lin"] = new_depth_range_lin

    print(f"depth ignore value: {max_depth}")
    print(f"depth range: {depth_range}")
    print(f"depth range warped (log): {new_depth_range_log}")
    print(f"depth range warped (lin): {new_depth_range_lin}")

    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=4)


if __name__ == '__main__':
    main()
