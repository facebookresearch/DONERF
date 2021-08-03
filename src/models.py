# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import torch

import torch.nn as nn
import torch.nn.functional as F

from util.config import Config
from collections import OrderedDict


class BaseNet(nn.Module):
    def __init__(self, D, W, skip, n_in, n_out, config, net_idx):
        super(BaseNet, self).__init__()
        self.net_idx = net_idx
        if "auto" in skip:

            skip_layer = 7

            if len(skip) > 4:
                skip_layer = int(skip[4:])

            if config.posEnc and config.posEnc[net_idx] and "RayMarch" in config.inFeatures[net_idx]:
                if config.posEnc[net_idx] == "nerf":
                    freq = config.posEncArgs[net_idx].split("-")
                    posInputs = (int(freq[0]))*6 + 3
                    dirInputs = (int(freq[1]))*6 + 2
                    # NOTE: this assumes 8 layers I guess
                    skip = f"0::{posInputs}-{D*skip_layer//8}:{posInputs}:"
            # did not override so set to nothing
            if "auto" in skip:
                print("Warning auto skip setup not detectable, using no skip connections")
                skip = ""
        self.name = f"relu{self.net_idx}({W}x{D}{skip.replace(':','.') if skip else ''})"
        self.D = D
        self.W = W
        self.inputLocations = {0:(0,n_in)}
        if skip:
            self.inputLocations = dict()
            decode_skips = [p for p in skip.split('-')]
            for s in decode_skips:
                match = re.search('^([0-9]+)(:?)([0-9]*)(:?)([0-9]*)$', s)
                if not match:
                    raise Exception("could not decode skip info")
                loc = match.group(1)
                has_first = match.group(2)
                start_feat = match.group(3)
                has_inbetween = match.group(4)
                end_feat = match.group(5)
                if has_first == '' and has_inbetween == '':
                    #all
                    self.inputLocations[int(loc)] = (0,n_in)
                elif has_first == ':' and has_inbetween == '':
                    single = int(start_feat+end_feat)
                    self.inputLocations[int(loc)] = (single, single+1)
                else:
                    istart = int(start_feat) if start_feat != '' else 0
                    iend =   int(end_feat) if end_feat != '' else n_in
                    self.inputLocations[int(loc)] = (istart, iend)

            if 0 not in self.inputLocations:
                self.inputLocations[0] = (0,n_in)
        self.n_in = n_in
        self.n_out = n_out
        layers = [nn.Linear(self.inputLocations[0][1]-self.inputLocations[0][0], self.W)]
        for i in range(1, self.D):
            layers.append(nn.Linear(self.inputLocations[i][1]-self.inputLocations[i][0] + self.W if i in self.inputLocations else self.W, self.W if i != self.D - 1 else self.n_out))
        # layers.append(nn.Linear(self.W, self.n_out))
        self.layers = nn.ModuleList(layers)

        self.activation = F.relu

        for i, l in enumerate(self.layers):
            nn.init.kaiming_normal_(l.weight)

        self.init_weights()

    def init_weights(self):
        pass

    def save_weights(self, path, name_suffix="", optimizer=None):
        torch.save(self.state_dict(), f"{path}{self.name}_{name_suffix}.weights")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), f"{path}{self.name}_{name_suffix}.optimizer")

    def delete_saved_weights(self, path, optimizer=None):
        ckpts = [os.path.join(path, f) for f in sorted(os.listdir(os.path.join(path))) if
                 '.weights' in f and self.name in f and not '_opt.weights' in f]
        # keep the last 10 files just in case something happened during training
        for file in ckpts[:-10]:
            # and also keep the weights every 50k iterations
            epoch = int(file.split('.weights')[0].split('_')[-1])
            if epoch % 50000 == 0 and epoch > 0:
                continue
            os.remove(file)
            if optimizer is not None:
                os.remove(f"{file.split('.weights')[0]}.optimizer")

    def load_weights(self, path, device):
        print('Reloading Checkpoint from', path)
        model = torch.load(path, map_location=device)
        # no idea why, but sometimes torch.load returns an ordered_dict...
        if type(model) == type(OrderedDict()):
            self.load_state_dict(model)
        else:
            self.load_state_dict(model.state_dict())

    def load_optimizer(self, path, optimizer, device):
        if os.path.exists(path):
            print(f"Reloading optimizer checkpoint from {path}")
            optimizer_state = torch.load(path, map_location=device)
            optimizer.load_state_dict(optimizer_state)

    def load_specific_weights(self, path, checkpoint_name, optimizer=None, device=0):
        ckpts = [os.path.join(path, f) for f in sorted(os.listdir(os.path.join(path))) if
                 checkpoint_name in f and self.name in f]
        if len(ckpts) == 0:
            print("no Checkpoints found")
            return 0

        ckpt_path = ckpts[-1]

        self.load_weights(ckpt_path, device)

        if optimizer is not None:
            optim_path = f"{ckpt_path.split('.weights')[0]}.optimizer"
            self.load_optimizer(optim_path, optimizer, device)
        return 1

    def load_latest_weights(self, path, optimizer=None, device=0, config=None):
        ckpts = [os.path.join(path, f) for f in sorted(os.listdir(os.path.join(path))) if
                 '.weights' in f and self.name in f and '_opt.weights' not in f]
        if len(ckpts) == 0:
            print("no Checkpoints found")
            if config and config.preTrained and len(config.preTrained) > self.net_idx and config.preTrained[self.net_idx].lower() != "none":
                wpath = os.path.join(config.preTrained[self.net_idx], f"{self.name}.weights")
                if os.path.exists(wpath):
                    print("loading pretrained weights")
                    self.load_weights(wpath, device)
                else:
                    print(f"WARNING pretrained weights not found: {wpath}")
                opath = wpath = os.path.join(config.preTrained[self.net_idx], f"{self.name}.optimizer")
                if optimizer is not None and os.path.exists(opath):
                    self.load_optimizer(opath, optimizer, device)
            return 0
        ckpt_path = ckpts[-1]

        try:
            epoch = int(ckpt_path.split('.weights')[0].split('_')[-1])
        except ValueError:
            epoch = 0

        self.load_weights(ckpt_path, device)

        if optimizer is not None:
            optim_path = f"{ckpt_path.split('.weights')[0]}.optimizer"
            self.load_optimizer(optim_path, optimizer, device)

        return epoch

    def forward(self, x):
        out = x[...,self.inputLocations[0][0]:self.inputLocations[0][1]]

        for i, l in enumerate(self.layers):
            if i in self.inputLocations and i != 0:
                out = torch.cat([out, x[...,self.inputLocations[i][0]:self.inputLocations[i][1]]], -1)
            out = l(out)

            if i + 1 < len(self.layers):
                # no activation for last layer
                out = self.activation(out)

        return out


# Model taken from nerf-pytorch
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, n_in=None, n_out=4, skips=[4], use_viewdirs=False, net_idx=None, config=None):
        """
        """

        super(NeRF, self).__init__()
        self.net_idx = net_idx
        self.D = D
        self.W = W
        self.skips = skips

        if 'auto' in skips[0]:
            self.skips = [4]
        else:
            self.skips = [int(x) for x in self.skips]

        self.name = f"NeRF{self.net_idx}({W}x{D}{self.skips})"
        self.input_ch = 3
        self.input_ch_views = 3
        self.output_ch = n_out

        if config.posEnc and config.posEnc[net_idx] and "RayMarch" in config.inFeatures[net_idx]:
            if config.posEnc[net_idx] == "nerf":
                freq = config.posEncArgs[net_idx].split("-")
                self.input_ch = (int(freq[0])) * 6 + 3
                self.input_ch_views = (int(freq[1])) * 6 + 3

        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, self.output_ch)

        for i, l in enumerate(self.pts_linears):
            nn.init.kaiming_normal_(l.weight)

        for i, l in enumerate(self.views_linears):
            nn.init.kaiming_normal_(l.weight)

        # self.init_weights()

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def save_weights(self, path, name_suffix="", optimizer=None):
        torch.save(self.state_dict(), f"{path}{self.name}_{name_suffix}.weights")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), f"{path}{self.name}_{name_suffix}.optimizer")

    def delete_saved_weights(self, path, optimizer=None):
        ckpts = [os.path.join(path, f) for f in sorted(os.listdir(os.path.join(path))) if
                 '.weights' in f and self.name in f and '_opt.weights' not in f]
        # keep the last 10 files just in case something happened during training
        for file in ckpts[:-10]:
            # and also keep the weights every 50k iterations
            epoch = int(file.split('.weights')[0].split('_')[-1])
            if epoch % 50000 == 0 and epoch > 0:
                continue
            os.remove(file)
            if optimizer is not None:
                os.remove(f"{file.split('.weights')[0]}.optimizer")

    def load_weights(self, path, device):
        print('Reloading Checkpoint from', path)
        model = torch.load(path, map_location=device)
        # no idea why, but sometimes torch.load returns an ordered_dict...
        if type(model) == type(OrderedDict()):
            self.load_state_dict(model)
        else:
            self.load_state_dict(model.state_dict())

    def load_optimizer(self, path, optimizer, device):
        if os.path.exists(path):
            print(f"Reloading optimizer checkpoint from {path}")
            optimizer_state = torch.load(path, map_location=device)
            optimizer.load_state_dict(optimizer_state)

    def load_specific_weights(self, path, checkpoint_name, optimizer=None, device=0):
        ckpts = [os.path.join(path, f) for f in sorted(os.listdir(os.path.join(path))) if
                 checkpoint_name in f and self.name in f]
        if len(ckpts) == 0:
            print("no Checkpoints found")
            return 0

        ckpt_path = ckpts[-1]

        self.load_weights(ckpt_path, device)

        if optimizer is not None:
            optim_path = f"{ckpt_path.split('.weights')[0]}.optimizer"
            self.load_optimizer(optim_path, optimizer, device)
        return 1

    def load_latest_weights(self, path, optimizer=None, device=0, config=None):
        ckpts = [os.path.join(path, f) for f in sorted(os.listdir(os.path.join(path))) if
                 '.weights' in f and self.name in f and not '_opt.weights' in f]
        if len(ckpts) == 0:
            print("no Checkpoints found")
            if config and config.preTrained and len(config.preTrained) > self.net_idx and config.preTrained[
                self.net_idx].lower() != "none":
                wpath = os.path.join(config.preTrained[self.net_idx], f"{self.name}.weights")
                if os.path.exists(wpath):
                    print("loading pretrained weights")
                    self.load_weights(wpath, device)
                else:
                    print(f"WARNING pretrained weights not found: {wpath}")
                opath = wpath = os.path.join(config.preTrained[self.net_idx], f"{self.name}.optimizer")
                if optimizer is not None and os.path.exists(opath):
                    self.load_optimizer(opath, optimizer, device)
            return 0
        ckpt_path = ckpts[-1]
        epoch = int(ckpt_path.split('.weights')[0].split('_')[-1])

        self.load_weights(ckpt_path, device)

        if optimizer is not None:
            optim_path = f"{ckpt_path.split('.weights')[0]}.optimizer"
            self.load_optimizer(optim_path, optimizer, device)

        return epoch


class ModelSelection:
    @classmethod
    def getModel(cls, config: Config, n_in, n_out, device, model_idx):
        i = model_idx
        if config.activation[i] == 'relu':
            return BaseNet(config.layers[i], config.layerWidth[i], config.skips[i], n_in, n_out, config, i).to(device)
        elif config.activation[i] == "nerf":
            return NeRF(config.layers[i], config.layerWidth[i], n_in=n_in, n_out=n_out, skips=[config.skips[i]], use_viewdirs=True, net_idx=i, config=config).to(device)
        else:
            raise Exception(f'Unknown activation {config.activation[i]}')
