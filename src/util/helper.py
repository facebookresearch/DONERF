# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch

import numpy as np


# torch to numpy
def t2np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
        a.device)
    return torch.index_select(a, dim, order_index)


def config_to_name(in_features, out_features, models, encodings, enc_args_in, loss_alpha, loss_beta):
    name = ""
    for i in range(len(in_features)):
        if i > 0:
            name += "_"
        enc_args = f"({enc_args_in[i]})" if enc_args_in[i] not in ["", "none"] else ""
        enc = f"({encodings[i]}{enc_args})" if encodings[i] not in ["", "none"] else ""

        loss_alpha_beta = ""

        if len(loss_alpha) > i and len(loss_beta) > i:
            loss_alpha_beta = f"l{loss_alpha[i]}_{loss_beta[i]}_"

        name += f"{loss_alpha_beta}{in_features[i].get_string()}{enc}-{models[i].name}-{out_features[i].get_string()}"
    return name
