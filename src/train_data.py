# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import torch

import numpy as np

from torch.utils.data import DataLoader
from datasets import DatasetInfo, worker_offset_sequence
from features import FeatureSet, FeatureSetKeyConstants
from models import BaseNet, ModelSelection
from util.feature_encoding import FeatureEncoding
from util.helper import config_to_name
from losses import get_loss_by_name
from typing import List
from importlib import import_module


class TrainConfig:
    def __init__(self):
        self.f_in: List[FeatureSet] = []
        self.f_out: List[FeatureSet] = []
        self.models: List[BaseNet] = []
        self.encodings: List[FeatureEncoding] = []
        self.enc_args = []
        self.optimizers = []
        self.losses = []
        self.loss_weights = []
        self.device: str = ""
        self.epoch0 = 0
        self.epochs = 300000
        self.logDir = ""
        self.config_file = None
        self.dataset_name = None
        self.experiment_name = None
        self.base_log_dir = ""
        self.pixel_idx_sequence_gen = None
        self.h = -1
        self.w = -1
        self.best_valid_loss = None
        self.best_valid_loss_pretrain = []

        self.dataset_info = None
        self.train_dataset = None
        self.train_data_loader = None
        self.pretrain_data_loader = None
        self.valid_dataset = None
        self.valid_data_loader = None
        self.test_dataset = None
        self.test_data_loader = None
        self.copy_to_gpu = None

        self.camera_type = None
        self.camera_path = None
        self.video_frames = None

    def initialize(self, config, load_data=True, log_path=None, training=True):
        self.config_file = config
        self.base_log_dir = self.config_file.logDir

        if config.randomSeed != -1:
            torch.manual_seed(config.randomSeed)
            np.random.seed(config.randomSeed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.device = f"cuda:{config.device}" if torch.cuda.is_available() else "cpu"

        print(f"device {torch.cuda.get_device_name(self.device)}")

        self.f_in, self.f_out = FeatureSet.get_sets(config, self.device)
        self.dataset_info = DatasetInfo(config, self)
        self.initialize_features()
        self.copy_to_gpu = not config.storeFullData

        self.models = []
        self.optimizers = []
        self.encodings = []
        self.enc_args = []
        self.losses = []
        self.loss_weights = []

        # Initialize list defaults per feature. This is so that the exported config file correctly contains
        # all the defaults which are otherwise initialized in the feature init code...
        if config.rayMarchSamplingNoise is None:
            config.rayMarchSamplingNoise = []

        if config.zNear is None:
            config.zNear = []

        if config.zFar is None:
            config.zFar = []

        for i in range(len(self.f_in)):
            model = ModelSelection.getModel(config, self.f_in[i].n_feat, self.f_out[i].n_feat, self.device, i)
            self.models.append(model)
            self.optimizers.append(torch.optim.Adam(model.parameters(), lr=config.lrate))
            self.encodings.append(config.posEnc[i])
            self.enc_args.append(config.posEncArgs[i])
            self.losses.append(get_loss_by_name(config.losses[i], config=config, net_idx=i))
            self.loss_weights.append(config.lossWeights[i])

            # Initialize list defaults per feature. This is so that the exported config file correctly contains
            # all the defaults which are otherwise initialized in the feature init code...
            if len(config.rayMarchSamplingNoise) <= i:
                config.rayMarchSamplingNoise.append(0.0)

            if len(config.zNear) <= i:
                config.zNear.append(0.001)

            if len(config.zFar) <= i:
                config.zFar.append(1.0)

            if hasattr(self.losses[i], 'requires_alpha_beta'):
                if len(config.lossAlpha) <= i:
                    config.lossAlpha.append(1.0)

                if len(config.lossBeta) <= i:
                    config.lossBeta.append(0.0)

        depth_transform = ""
        if config.depthTransform and config.depthTransform != "linear":
            depth_transform = config.depthTransform[0:2] + "_"

        scale_interpolation = ""
        if config.scaleInterpolation and config.scaleInterpolation != "median":
            scale_interpolation = config.scaleInterpolation[0:2] + "_"

        experiment_name = depth_transform + scale_interpolation + \
                          config_to_name(self.f_in, self.f_out, self.models, self.encodings, self.enc_args, config.lossAlpha, config.lossBeta)

        dataset_name = os.path.basename(os.path.normpath(config.data)) + "/"
        if log_path is None:
            self.logDir = os.path.join(config.logDir, dataset_name, experiment_name) + "/"
            self.dataset_name = dataset_name
            self.experiment_name = experiment_name
        else:
            self.logDir = log_path

        # just to prevent bugs
        self.config_file.logDir = self.logDir
        os.makedirs(config.logDir, exist_ok=True)
        os.makedirs(f"{config.logDir}/test_opt/", exist_ok=True)
        self.epochs = self.config_file.epochs

        # load previous best validation loss (if any)
        if os.path.exists(os.path.join(config.logDir, "opt.txt")):
            with open(os.path.join(config.logDir, "opt.txt")) as f:
                line = f.readline()
                match = re.search(r'\d+\.\d+', line)
                self.best_valid_loss = float(line[match.start():match.end()])

        for i in range(len(self.models)):
            if os.path.exists(os.path.join(config.logDir, f"opt_{i}.txt")):
                with open(os.path.join(config.logDir, f"opt_{i}.txt")) as f:
                    line = f.readline()
                    match = re.search(r'\d+\.\d+', line)
                    self.best_valid_loss_pretrain.append(float(line[match.start():match.end()]))

        if not os.path.exists(os.path.join(config.logDir, "config.ini")):
            # Copy config params (including command line params) by serializing dict again

            # This is used to replace the quotes
            translation = {39: None}

            with open(os.path.join(config.logDir, "config.ini"), 'w') as f:
                for key in self.config_file.__dict__:
                    val = self.config_file.__dict__[key]

                    if val is not None:
                        # Skip empty lists
                        if isinstance(val, list) and len(val) == 0:
                            continue

                        f.write(f'{key} = {str(self.config_file.__dict__[key]).translate(translation)}\n')

        if load_data:
            self.pixel_idx_sequence_gen = getattr(import_module("sampled_sequence"), self.config_file.sampleGenerator)\
                (dims=2, device='cpu', base_log_dir=self.base_log_dir, num_pregeneration=30000000)

            if config.storeFullData:
                from datasets import FullyLoadedViewCellDataset as Dataset
                num_workers = 0
                pin_memory = False
                worker_init_fn = None
            else:
                from datasets import OnTheFlyViewCellDataset as Dataset
                num_workers = config.numWorkers
                pin_memory = True
                worker_init_fn = worker_offset_sequence

            if training:
                self.train_dataset = Dataset(self.config_file, self, self.dataset_info, set_name="train",
                                             num_samples=self.config_file.samples)
                self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.config_file.batchImages,
                                                    shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
                                                    persistent_workers=pin_memory, worker_init_fn=worker_init_fn)

                self.valid_dataset = Dataset(self.config_file, self, self.dataset_info, set_name="val",
                                             num_samples=self.config_file.samples)
                self.valid_data_loader = DataLoader(self.valid_dataset, batch_size=self.config_file.batchImages,
                                                    shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
                                                    persistent_workers=pin_memory, worker_init_fn=worker_init_fn)

                # we create another DataLoader here, so we can use a different number of images for pretraining
                # num_samples can simply be changed in the dataset, but batch size appears to be unchangeable after
                # creation
                if self.config_file.epochsPretrain is not None and len(self.config_file.epochsPretrain) != 0 and \
                        self.config_file.batchImagesPretrain != -1:
                    self.pretrain_data_loader = DataLoader(self.train_dataset,
                                                           batch_size=self.config_file.batchImagesPretrain,
                                                           shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
                                                           persistent_workers=pin_memory, worker_init_fn=worker_init_fn)

            self.test_dataset = Dataset(self.config_file, self, self.dataset_info, set_name="test",
                                        num_samples=self.dataset_info.w * self.dataset_info.h)
            self.test_data_loader = DataLoader(self.test_dataset, batch_size=1,
                                               shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
                                               persistent_workers=pin_memory, worker_init_fn=worker_init_fn)

    # this is only used for exporting after evaluate.py has been used, as training dataset is not loaded there
    def import_train_dataset(self):
        if self.config_file.storeFullData:
            from datasets import FullyLoadedViewCellDataset as Dataset
        else:
            from datasets import OnTheFlyViewCellDataset as Dataset

        self.train_dataset = Dataset(self.config_file, self, self.dataset_info, set_name="train",
                                     num_samples=self.config_file.samples)

    def initialize_features(self):
        for f in self.f_in:
            f.initialize(self.config_file, self.dataset_info, self.device)
        for f in self.f_out:
            f.initialize(self.config_file, self.dataset_info, self.device)

    def get_data_set_and_loader(self, dataset_name):
        if dataset_name == "train":
            return self.train_dataset, self.train_data_loader
        elif dataset_name == "val":
            return self.valid_dataset, self.valid_data_loader
        else:
            return self.test_dataset, self.test_data_loader

    def delete_data_sets_and_laoders(self):
        if self.pretrain_data_loader is not None:
            del self.pretrain_data_loader
        if self.train_data_loader is not None:
            del self.train_data_loader
            del self.train_dataset
        if self.valid_data_loader is not None:
            del self.valid_data_loader
            del self.valid_dataset
        if self.test_data_loader is not None:
            del self.test_data_loader
            del self.test_dataset

    def inference(self, batch_idx, gradient=False, **kwargs):
        postprocessed_outs = []
        inference_dicts = []

        for i in range(len(self.models)):
            batch_input = batch_idx.get_batch_input(i)

            # We reshape this to [-1, N_feat] to enable splitting it into chunks to save memory
            inference_dict = self.f_in[i].batch(batch_input, prev_outs=inference_dicts, **kwargs)
            model_input = inference_dict[FeatureSetKeyConstants.input_feature_batch]
            if gradient:
                inference_dict[FeatureSetKeyConstants.network_output] = self.models[i](model_input)
            else:
                with torch.no_grad():
                    inference_dict[FeatureSetKeyConstants.network_output] = self.models[i](model_input)

            self.f_in[i].postprocess(inference_dict, batch_input)
            postprocessed_outs.append(inference_dict[FeatureSetKeyConstants.postprocessed_network_output])
            inference_dicts.append(inference_dict)

        # For computing the loss or rendering images, we want the postprocessed output
        return postprocessed_outs, inference_dicts

    def store_camera_options(self):
        if self.camera_type is not None:
            print(f"Warning: there are saved camera options - keeping previously saved values!")
            return

        self.camera_type = self.config_file.camType
        self.camera_path = self.config_file.camPath
        self.video_frames = self.config_file.videoFrames

    def restore_camera_options(self):
        if self.camera_type is None:
            print(f"Warning: there are no saved camera options!")
        else:
            self.config_file.camType = self.camera_type
            self.config_file.camPath = self.camera_path
            self.config_file.videoFrames = self.video_frames

            self.camera_type = None
            self.camera_path = None
            self.video_frames = None

    def save_weights(self, name_suffix, model_idx=-1):
        for i, model in enumerate(self.models):
            if model_idx == -1 or model_idx == i:
                model.delete_saved_weights(self.logDir, optimizer=self.optimizers[i])
                model.save_weights(self.logDir, name_suffix=name_suffix, optimizer=self.optimizers[i])

    def load_latest_weights(self):
        for i, model in enumerate(self.models):
            self.epoch0 = model.load_latest_weights(self.logDir, self.optimizers[i], self.device, self.config_file) + 1

    def load_specific_weights(self, name, model_idx=-1):
        for i, model in enumerate(self.models):
            if model_idx == -1 or model_idx == i:
                model.load_specific_weights(self.logDir, name, self.optimizers[i], self.device)

    def weights_locked(self, epoch, net_idx):
        e_bef = -1
        e_aft = -1

        if self.config_file.epochsLockWeightsBefore is not None and len(
                self.config_file.epochsLockWeightsBefore) > net_idx:
            e_bef = self.config_file.epochsLockWeightsBefore[net_idx]

        if self.config_file.epochsLockWeightsAfter is not None and len(
                self.config_file.epochsLockWeightsAfter) > net_idx:
            e_aft = self.config_file.epochsLockWeightsAfter[net_idx]

        if e_bef == -1 and e_aft != -1:
            # weights locked after e_aft
            return epoch > e_aft
        elif e_bef != -1 and e_aft == -1:
            # weights locked before e_bef
            return epoch < e_bef
        elif e_bef != -1 and e_aft != -1:
            # weights locked between e_aft and e_bef (e_aft < e_bef)
            return e_bef > epoch > e_aft

        return False
