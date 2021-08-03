# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from datasets import create_sample_wrapper
from util.config import Config
from train_data import TrainConfig
from plots import render_img, render_all_imgs, render_video


def main():
    config = Config.init(only_known_args=True)
    train_config = TrainConfig()
    train_config.initialize(config)

    train_config.load_latest_weights()

    if config.checkPointName is not None and config.checkPointName != "":
        train_config.load_specific_weights(config.checkPointName)

    # NOTE: if using another dataset to render_img (valid, train), need to "dataset.full_images" to True first
    #       (see render_all_imgs in plots.py)
    img_samples = create_sample_wrapper(train_config.test_dataset[0], train_config, True)
    render_img(train_config, img_samples, img_name=f"test_0")

    render_all_imgs(train_config, "test_images/", dataset_name="train")
    render_all_imgs(train_config, "test_images/", dataset_name="val")
    render_all_imgs(train_config, "test_images/", dataset_name="test")

    render_video(train_config, vid_name=config.outputVideoName)


if __name__ == '__main__':
    main()
