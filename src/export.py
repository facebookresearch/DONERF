# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os

from datasets import create_sample_wrapper
from train_data import TrainConfig
from util.config import Config


def write_pos_enc(n_freqs, f, path, device):

    if os.path.exists(path):
        min, max = torch.load(path, device)

        f.write(str(min.cpu().numpy()) + "\n")
        f.write(str(max.cpu().numpy()) + "\n")
    max_freq = n_freqs - 1
    freq_bands = 2. ** torch.linspace(0., max_freq, steps=n_freqs)
    for frq in freq_bands:
        f.write(str(frq.cpu().numpy()) + "\n")


def export_onnx(train_config=None, out_dir=None):
    print("exporting onnx started ... ")

    if train_config is None:
        config = Config.init()
        train_config = TrainConfig()
        train_config.initialize(config)
        if config.checkPointName is not None and config.checkPointName != "":
            train_config.load_specific_weights(config.checkPointName)
        else:
            train_config.load_latest_weights()
    else:
        config = train_config.config_file

    if out_dir is None:
        out_dir = config.logDir
    else:
        os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/dataset_info.txt", "w") as f:
        f.write("view_cell_center = " + str(train_config.dataset_info.view.view_cell_center) + "\n")
        f.write("view_cell_size = " + str(train_config.dataset_info.view.view_cell_size) + "\n")
        f.write("depth_range = " + str(train_config.dataset_info.depth_range) + "\n")
        f.write("fov = " + str(train_config.dataset_info.view.fov) + "\n")
        f.write("focal = " + str(train_config.dataset_info.view.focal) + "\n")
        f.write("camera_scale = " + str(train_config.dataset_info.view.camera_scale) + "\n")
        f.write("max_depth = " + str(train_config.dataset_info.depth_max) + "\n")

    if train_config.train_dataset is None:
        train_config.import_train_dataset()

    img_samples = create_sample_wrapper(train_config.train_dataset[0], train_config, True)
    input_names = ["input_1"]
    output_names = ["output1"]

    for b1 in img_samples.batches(128):
        _, inference_dicts = train_config.inference(b1)

        input_feature_batches = []

        for i in range(len(inference_dicts)):
            input_feature_batches.append(inference_dicts[i]['InputFeatureBatch'])

        del inference_dicts
        torch.cuda.empty_cache()

        for model_idx in range(0, len(train_config.models)):
            m = train_config.models[model_idx]

            torch.onnx.export(m, input_feature_batches[model_idx], f"{out_dir}/model{model_idx}.onnx", verbose=True,
                              export_params=True, input_names=input_names, output_names=output_names,
                              dynamic_axes={'input_1': {0: '-1'}, 'output1': {0: '-1'}})

            b_ = input_feature_batches[model_idx].cpu().numpy()
            f = open(f"{out_dir}/feature_sample.txt", "w")

            batch = input_feature_batches[model_idx]
            input_feature_batches[model_idx] = None
            del batch
            torch.cuda.empty_cache()

        break


if __name__ == "__main__":
    export_onnx()
