# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from util.helper import t2np, tile
from matplotlib import pyplot as plt


def transform_img(img, dim):
    if img.shape[-1] == 1:
        img = tile(img, dim=len(img.shape) - 1, n_tile=3)

    img = t2np(img)

    if img.shape[-1] < 3:
        new_shape = list(img.shape)
        new_shape[-1] = 3
        new_img = np.zeros(new_shape, dtype=np.single)
        for i in range(min(3, img.shape[-1])):
            new_img[..., i] = img[..., i]
        img = new_img

    if img.shape[-1] == dim.h * dim.w:
        maxim = img.max() + 1
        step = 1. / maxim
        img = 0.5 * step + img * step
        img = np.repeat(img[:, np.newaxis], 3, axis=1)

    # multi depth to single depth
    if img.shape[-1] > 4:
        step = 1. / img.shape[-1]
        ids = np.argsort(img)
        ids = ids[..., -3:]
        r = range(dim.h * dim.w)
        new_img = np.zeros((dim.h * dim.w, 3))
        min_val = np.amin(img)
        for i in range(3):
            mask = img[r, ids[:, i]] > min_val
            new_img[mask, i] = 0.5 * step + ids[mask, i] * step
        img = new_img

    return np.clip(img.reshape(dim.h, dim.w, -1)[:, :, :3], 0., 1.)


def save_img(img, dim, path):
    if path is None:
        return

    img = transform_img(img, dim)

    plt.imsave(path, img)
