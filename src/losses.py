# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch


class LimitedDepthMSELoss(torch.nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str='mean', config=None, net_idx=-1) -> None:
        super(LimitedDepthMSELoss, self).__init__()
        self.reduction = reduction
        self.mseLoss = torch.nn.MSELoss()
        self.ignore_value = config.multiDepthIgnoreValue[net_idx]

    def forward(self, outputs: torch.Tensor, targets : torch.Tensor) -> torch.Tensor:
        seltargets = torch.where(targets.data < self.ignore_value, targets.data, outputs.data)
        return self.mseLoss(outputs, seltargets)


class MultiDepthLimitedMSELoss(torch.nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str='mean', config=None, net_idx=-1) -> None:
        super(MultiDepthLimitedMSELoss, self).__init__()
        self.reduction = reduction
        self.mseLoss = torch.nn.MSELoss()
        self.ignore_value = config.multiDepthIgnoreValue[net_idx]

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # copy outputs
        outputs_cpy = outputs.clone()
        resort_indices = torch.zeros_like(outputs, dtype=torch.int64)

        # compute minimum distance from predictions to target
        # rep_targets = torch.repeat_interleave(targets, targets.shape[-1], dim=-1)
        for i in range(targets.shape[-1]):
            # sel_targets = rep_targets[:,3*i:3*(i+1)]
            sel_targets = torch.reshape(targets[..., i], (targets.shape[0], 1)).repeat(1, targets.shape[-1])
            diff2 = torch.abs(outputs_cpy - sel_targets)
            ids = torch.argmin(diff2, -1)
            outputs_cpy = outputs_cpy.scatter_(1, torch.reshape(ids, (targets.shape[0], 1)), torch.finfo(outputs.dtype).max)
            # outputs_cpy[..., ids] = torch.finfo(outputs.dtype).max
            resort_indices[..., i] = ids
        outputs_shfled = torch.gather(outputs, 1, resort_indices)

        seltargets = torch.where(targets.data != self.ignore_value, targets.data, outputs_shfled.data)
        return self.mseLoss(outputs_shfled, seltargets)


class MSEPlusWeightAccum(torch.nn.Module):

    def __init__(self, config=None, net_idx=-1) -> None:
        super(MSEPlusWeightAccum, self).__init__()
        self.mseLoss = torch.nn.MSELoss()
        self.asymmetric = True
        self.loss_alpha = config.lossAlpha[net_idx]
        self.loss_beta = config.lossBeta[net_idx]
        self.requires_alpha_beta = True

    def forward(self, outputs: torch.Tensor, targets : torch.Tensor, **kwargs) -> torch.Tensor:
        inference_dict = kwargs.get('inference_dict', None)

        if inference_dict is None:
            raise Exception(f"MSEPlusWeightAccum requires inference_dict argument!")

        weights_sum = torch.sum(inference_dict['NeRFWeightsOutput'], axis=1)

        loss_mse = self.mseLoss(outputs, targets)
        alpha = self.loss_alpha
        beta = self.loss_beta

        # The weights should sum to >= 1.0

        if self.asymmetric:
            weights_sum[weights_sum > 1.0] = 1.0

        loss_weights = self.mseLoss(weights_sum, torch.ones_like(weights_sum))

        return alpha * loss_mse + beta * loss_weights


class DefaultLossWrapper(torch.nn.Module):
    def __init__(self, loss_func) -> None:
        super(DefaultLossWrapper, self).__init__()
        self.loss_func = loss_func

    def forward(self, outputs: torch.Tensor, targets : torch.Tensor, **kwargs) -> torch.Tensor:
        return self.loss_func(outputs, targets)


def get_loss_by_name(name, config, net_idx):
    if name == "MSE":
        return DefaultLossWrapper(torch.nn.MSELoss())
    if name == "LimitedDepthMSE":
        return DefaultLossWrapper(LimitedDepthMSELoss(config=config, net_idx=net_idx))
    if name == "MultiDepthLimitedMSE":
        return DefaultLossWrapper(MultiDepthLimitedMSELoss(config=config, net_idx=net_idx))
    if name == "MSEPlusWeightAccum":
        return MSEPlusWeightAccum(config=config, net_idx=net_idx)
    if name == "BCEWithLogitsLoss":
        return DefaultLossWrapper(torch.nn.BCEWithLogitsLoss())
    if name == "CrossEntropyLoss":
        return DefaultLossWrapper(torch.nn.CrossEntropyLoss())
    if name == "CrossEntropyLossWeighted":
        weights = torch.ones(config.multiDepthFeatures[net_idx] + 1, dtype=torch.float32).cuda(device=config.device)
        weights[-1] = 0.
        return DefaultLossWrapper(torch.nn.CrossEntropyLoss(weight=weights))
    if name.lower() == "none":
        return None
    else:
        raise Exception(f"Loss {name} unknown")
