# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch


class FeatureEncoding:
    name = "feat_enc"

    def __init__(self, config=None, name_suffix=""):
        # can't use path directly, because this is set afterwards
        self.config = config
        self.name_suffix = name_suffix
        self.encode_fns = None

    def initialize(self, *args, **kwargs):
        pass

    def encode(self, x, *args, **kwargs):
        return torch.cat([fn(x) for fn in self.encode_fns], - 1)

    @classmethod
    def get_encoding(cls, name):
        if name == "nerf":
            return PositionalEncoding
        if name == "none":
            return NoEncoding
        raise Exception(f"Encoding {name} not implemented")

    @classmethod
    def num_features(cls, name, n, n_freqs):
        if name == "nerf":
            return n * 2 * n_freqs + n
        if name == "none":
            return n


class NoEncoding(FeatureEncoding):
    name = "no_enc"

    def __init__(self, config=None, name_suffix=""):
        super().__init__(config, name_suffix)

    def initialize(self, *args, **kwargs):
        pass

    def encode(self, x, *args, **kwargs):
        return x


class PositionalEncoding(FeatureEncoding):
    name = "pos_enc"

    def __init__(self, config=None, name_suffix=""):
        super().__init__(config, name_suffix)

    def initialize(self, *args, **kwargs):
        n_freqs = kwargs["n_freqs"]

        max_freq = n_freqs - 1
        freq_bands = 2. ** torch.linspace(0., max_freq, steps=n_freqs)
        periodic_fns = [torch.sin, torch.cos]
        encode_fns = [lambda a: a]

        # nerf version
        for freq in freq_bands:
            for p_fn in periodic_fns:
                encode_fns.append(lambda a, p_fn=p_fn, freq=freq: p_fn(a * freq))

        self.encode_fns = encode_fns
