# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os

import numpy as np


"""
This class references / takes code from the excellent Stackoverflow answer / blog post of Martin Roberts
about low discrepancy sampling sequences. We use this to evenly spread the random samples in images space for
our training batches.
https://stats.stackexchange.com/a/355208
http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
"""

class RSequenceGenerator:

    def phi(self, d):
        x = 2.0000
        for i in range(10):
            x = pow(1 + x, 1 / (d + 1))
        return x

    def __init__(self, dims, seed=0.5):
        self.dims = dims
        self.seed = seed
        self.g = self.phi(dims)
        self.alpha = np.zeros(self.dims)
        self.sequence_index = 0
        for j in range(self.dims):
            self.alpha[j] = pow(1 / self.g, j + 1) % 1

    def generate_next_float(self):
        val = (self.seed + self.alpha * (self.sequence_index + 1)) % 1
        self.sequence_index += 1
        return val

    def generate_next_discrete(self, minv=0, maxv=400):
        val = (self.seed + self.alpha * (self.sequence_index + 1)) % 1

        value_range = maxv - minv
        discrete_val = (np.floor(value_range * val)).astype(int) + minv
        self.sequence_index += 1
        return discrete_val

    def generate_unique_discrete_array(self, num_elements, minv=0, maxv=400):
        ret_set = set()

        while len(ret_set) < num_elements:
            ret_set.add(tuple(self.generate_next_discrete(minv=minv, maxv=maxv)))

        return np.array(list(ret_set))


class PreGeneratedRSequenceGenerator:

    def phi(self, d):
        x = 2.0000
        for i in range(10):
            x = pow(1 + x, 1 / (d + 1))
        return x

    def load_state(self, *args, **kwargs):
        if os.path.exists(self.get_path()):
            return torch.load(self.get_path(), map_location=kwargs["device"])
        return None

    def save_state(self, state, *args, **kwargs):
        torch.save(state, self.get_path())

    def get_path(self):
        return os.path.join(self.base_log_dir, f"{self.name}_{self.name_suffix}.tar")

    def __init__(self, dims, device, base_log_dir, num_pregeneration=30000000, seed=0.5):
        self.dims = dims
        self.seed = seed
        self.g = self.phi(dims)
        self.alpha = np.zeros(self.dims)
        self.sequence_index = 0
        self.device = device
        self.base_log_dir = base_log_dir
        self.name = 'PreGeneratedRSequenceGenerator'
        self.name_suffix = '{0}_{1}'.format(dims, num_pregeneration)

        self.offset_start = 0

        for j in range(self.dims):
            self.alpha[j] = pow(1 / self.g, j + 1) % 1

        self.pregenerated_tensor = self.load_state(device=device)

        # Pregenerate tons of random values and put them as torch tensors on the gpu
        if self.pregenerated_tensor is None:
            print('\nPreprocessing low-discrepancy r sequence values for sampling')
            pregenerated_values = []
            for i in range(num_pregeneration):
                pregenerated_values.append(self.generate_next_float())
                if i % 1000 == 0:
                    print(f"\r{i + 1}/{num_pregeneration}", end="")

            self.pregenerated_tensor = torch.tensor(pregenerated_values, device=device, dtype=torch.float32)
            self.save_state(self.pregenerated_tensor)

    def generate_next_float(self):
        val = (self.seed + self.alpha * (self.sequence_index + 1)) % 1
        self.sequence_index += 1
        return val

    def get_discrete_tensor_subset(self, num_elements, device, minv=0, maxv=400):

        offset_end = self.offset_start + num_elements

        if offset_end > len(self.pregenerated_tensor):
            offset_end = num_elements
            self.offset_start = 0

        value_range = maxv - minv

        # * 0.99999 is to make sure that this function never returns max (is exclusive) no matter what the underlying initial tensor has in it.
        tensor_subset = torch.floor(self.pregenerated_tensor[self.offset_start:offset_end].to(device) * value_range * 0.99999).long() + minv

        self.offset_start = offset_end

        return tensor_subset

    def set_offset(self, worker_id):
        self.offset_start = worker_id


class PreGeneratedUniformRandomSequenceGenerator(PreGeneratedRSequenceGenerator):

    def __init__(self, dims, device, base_log_dir, num_pregeneration=30000000, seed=0.5):
        self.dims = dims
        self.device = device
        self.base_log_dir = base_log_dir
        self.name = 'PreGeneratedUniformRandomSequenceGenerator'
        self.name_suffix = '{0}_{1}'.format(dims, num_pregeneration)

        self.offset_start = 0

        self.pregenerated_tensor = self.load_state(device=device)

        # Pregenerate tons of random values and put them as torch tensors on the gpu
        if self.pregenerated_tensor is None:
            print('Preprocessing uniform random sequence values for sampling')
            self.pregenerated_tensor = torch.rand([num_pregeneration, dims], device=device, dtype=torch.float32)
            self.save_state(self.pregenerated_tensor)
