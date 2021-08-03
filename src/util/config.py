# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import configargparse

from abc import ABC


class Config(ABC):
    _parser = None

    @classmethod
    def init(cls, path=None, only_known_args=False):
        if cls._parser is not None:
            return cls._parser

        cls._parser = configargparse.ArgParser()
        p = cls._parser
        p.add_argument('-c', '--config', is_config_file=True)

        # Data params
        p.add_argument('-data', '--data', required=True, type=str)
        p.add_argument('-log', '--logDir', required=True, type=str)
        p.add_argument('--trainStatsName', default='logs.csv', type=str)
        p.add_argument('--preTrained', default=[], action='append', type=str,
                       help="load pretrained weights (and encoding)")
        p.add_argument('--depthTransform', default="linear", type=str,
                       help="depth transform [log, linear], default = linear",  choices=["log", "linear"])
        p.add_argument('-s', '--scale', default=2, type=int,
                       help="scale factor for image (4 means reduction to fourth of size)")
        p.add_argument('--scaleInterpolation', default="median", type=str,
                       help="how the depth data is interpolated when downsampling with scale",
                       choices=["area", "leaveOut", "median"])

        # Feature params
        p.add_argument('-if', '--inFeatures', default=[], action='append', type=str,
                       choices=["SpherePosDir", "RayMarchFromPoses", "RayMarchFromCoarse"])
        p.add_argument('-of', '--outFeatures', default=[], action='append', type=str,
                       choices=["ClassifiedDepth", "RGBARayMarch"])
        p.add_argument('-pe', '--posEnc', default=[], action='append', type=str, choices=["none", "nerf"])
        p.add_argument('--posEncArgs', default=[], type=str, action='append',
                       help="special parameters for positional encoding")
        p.add_argument('--raySampleInput', default=[], type=int, action='append',
                       help="Add additional inputs for sample locations along the ray (supported in SpherePosDir).")

        # Network params
        p.add_argument('-act', '--activation', default=[], type=str, action='append',
                       choices=["relu", "nerf"])
        p.add_argument('-l', '--layers', default=[], type=int, action='append', help="number of layers")
        p.add_argument('-lw', '--layerWidth', default=[], type=int, action='append', help="width of each layer")
        p.add_argument('-sk', '--skips', default=[], type=str, action='append', help="skip information")

        # Training params
        p.add_argument('-d', '--device', default=0, type=int, help="Which cuda device to use")
        p.add_argument('-e', '--epochs', default=300001, type=int, help="training epochs")
        p.add_argument('--batchImages', default=-1, type=int,
                       help="Number of images used per batch. Set to -1 to use all.")
        p.add_argument('-smpl', '--samples', default=128, type=int, help="random sample per image for each epoch")
        p.add_argument('--lrate', default=0.0001, type=float,
                       help="Learning rate used in optimizer")
        p.add_argument('--lrate_decay', default=0.1, type=float,
                       help="Learning rate decay gamma used in optimizer")
        p.add_argument('--lrate_decay_steps', default=300000, type=int,
                       help="How many steps until the lrate_decay parameter is fully applied")
        p.add_argument('--losses', default=[], type=str,
                       choices=["none", "None", "MSE", "LimitedDepthMSE",
                                "MultiDepthLimitedMSE", "BCEWithLogitsLoss", "CrossEntropyLoss",
                                "CrossEntropyLossWeighted", "MSEPlusWeightAccum"],
                       action='append', help="losses for each output of the network")
        p.add_argument('--lossWeights', default=[], type=float, action='append',
                       help="loss weights for each output of the network")
        p.add_argument('--lossAlpha', default=[], type=float, action='append',
                       help="loss alpha for combined losses")
        p.add_argument('--lossBeta', default=[], type=float, action='append',
                       help="loss beta for combined losses")
        p.add_argument('-r', '--randomSeed', default=-1, type=int,
                       help="Random seed for numpy and pytorch. Set to -1 to use random seed.")
        p.add_argument('--sampleGenerator', default="PreGeneratedRSequenceGenerator", type=str,
                       choices=["PreGeneratedRSequenceGenerator", "PreGeneratedUniformRandomSequenceGenerator"])
        p.add_argument('--storeFullData', default=False, action="store_true",
                       help="Store full feature data (images, etc.) on the GPU, or only batch-per-batch data.")
        p.add_argument("--numWorkers", default=8, type=int, help="Number of workers used for runtime data loading")

        # PreTraining params
        p.add_argument('--epochsPretrain', default=[], type=int, action='append',
                       help="pretrain each network on GT until epoch x. Set to -1 to skip pretraining")
        p.add_argument('--batchImagesPretrain', default=-1, type=int,
                       help="number of images used per batch for Pretraining. Set to -1 to use batchImages")
        p.add_argument('--samplesPretrain', default=-1, type=int,
                       help="random sample per image for each epoch for pretraining, set to -1 to use samples")
        p.add_argument('--epochsLockWeightsBefore', default=[], type=int, action='append',
                       help="weights will not be updated before epoch X. -1 if never locked")
        p.add_argument('--epochsLockWeightsAfter', default=[], type=int, action='append',
                       help="weights will not be updated after epoch X. -1 if never locked")

        # Training Output params
        p.add_argument('-Eckpt', '--epochsCheckpoint', default=10000, type=int, help="save checkpoint after epochs")
        p.add_argument('-Er', '--epochsRender', default=10000, type=int, help="render image after epochs")
        p.add_argument('-Ev', '--epochsValidate', default=50000, type=int, help="validate model after epochs")
        p.add_argument('--epochsVideo', default=-1, type=int,
                       help="render video after epochs -> -1 means no video is rendered during training")
        p.add_argument('--videoFrames', default=-1, type=int, help="number of images render per video (-1 means all)")
        p.add_argument('--inferenceChunkSize', default=65536, type=int,
                       help="split number of inference inputs into chunks of this size to save memory")
        p.add_argument("-nV", "--nonVerbose", default=False, action="store_true", help="does not print epoch, loss and "
                                                                                       "PSNR to tqdm bar if set")

        # NeRF/Raymarching-params
        p.add_argument("--zNear", default=[], type=float, action='append',
                       help="z near for raymarch-based feature sets, such as RayMarchFromPoses. NeRF uses 2 for Lego.")
        p.add_argument("--zFar", default=[], type=float, action='append',
                       help="z far for raymarch-based feature sets, such as RayMarchFromPoses. NeRF uses 6 for Lego")
        p.add_argument("--numRaymarchSamples", default=[], type=int, action='append',
                       help="number of samples for raymarch-based feature sets, such as RayMarchFromPoses")
        p.add_argument("--rayMarchSampler", default=[], type=str, action='append',
                       choices=["none", "LinearlySpacedZNearZFar", "LinearlySpacedFromDepth",
                                "LinearlySpacedFromMultiDepth", "FromClassifiedDepth"])
        p.add_argument("--deterministicSampling", default=False, action="store_true",
                       help="whether or not to use deterministic ray sampling")
        p.add_argument("--rayMarchSamplingStep", default=[], type=float, action='append')
        p.add_argument("--rayMarchSamplingNoise", default=[], type=float, action='append')
        p.add_argument('--trainWithGTDepth', default=False, action="store_true")
        p.add_argument("--rayMarchNormalization", default=[], type=str, action='append',
                       help="way to normalize coordinates before feature encoding, default is MaxDepth to transform "
                            "coordinates in a <1 range",
                       choices=["None", "Centered", "MaxDepth", "MaxDepthCentered", "LogCentered",
                                "InverseDistCentered", "InverseSqrtDistCentered"])
        p.add_argument("--rayMarchNormalizationCenter", default=[], type=float, action='append',
                       help='Use this center instead of view cell center')
        p.add_argument("--perturb", default=False, action="store_true",
                       help="Whether or not to jitter samples during training. 0 -> no jittering.")

        # Video camera params
        p.add_argument("--camType", default="PredefinedCamera", type=str,
                       choices=["CenteredCamera", "RotatingCamera", "TranslatingCamera", "PredefinedCamera",
                                "ViewCellForwardCamera"])
        p.add_argument("--camCenter", default=[], type=float, action='append', help="center of camera")
        p.add_argument("--camRadius", default=4, type=float, help="radius around 0,0,0")
        p.add_argument("--camUpAngle", default=20, type=float, help="angle around up axis")
        p.add_argument("--camRightAngle", default=20, type=float, help="angle around right axis")
        p.add_argument("--movementVector", default=[], type=float, action='append',
                       help="movement vector for translating camera")
        p.add_argument('--camPath', default='cam_path_pan', type=str)

        # Test params
        p.add_argument("--checkPointName", default="opt.weights", type=str, help="checkpoint name to use for test.py")
        p.add_argument("--outputNetworkRaw", default=[], type=str, action='append',
                       help="Store the raw inference_dict outputs of the network for each element in this list")
        p.add_argument("--outputVideoName", default="test_video", type=str, help="output video name for test.py")

        # Multi Depth params
        p.add_argument("--multiDepthFeatures", default=[], action='append', type=int,
                       help="number of depth features produced for multi depth features")
        p.add_argument("--multiDepthWindowSize", default=[], action='append', type=str,
                       help="window size for multi depth generation")
        p.add_argument("--multiDepthIgnoreValue", default=[], action='append', type=float,
                       help="value to encode empty pixel")

        # Evaluation params
        p.add_argument("--performEvaluation", default=False, action="store_true",
                       help="performs evaluation of the trained net after training")

        if only_known_args is True:
            if path is None:
                args, unknown = p.parse_known_args()
                return args
            else:
                args, unknown = p.parse_known_args(['-c', path])
                return args

        if path is None:
            return p.parse_args()
        else:
            return p.parse_args(['-c', path])
