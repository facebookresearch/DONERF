# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import configargparse
import torch
import imageio
import cv2
import re
import gc
import platform

import numpy as np

from train_data import TrainConfig
from datasets import CameraViewCellDataset, create_sample_wrapper
from ptflops import flops_counter
from util.config import Config
from util.flip_loss import FLIP
from plots import save_img, render_video
from matplotlib import cm
from util.helper import t2np
from PIL import Image
from shutil import copyfile
from tqdm import tqdm
from export import export_onnx


class QualityContainer:
    def __init__(self):
        self.flip = []
        self.mse = []
        self.psnr = []
        self.ssim = []

        self.diff_data = None
        self.square_diff_data = None
        self.flip_data = None
        self.out_data = None


def calculate_mse(diff):
    return diff.pow(2).sum() / (diff.numel())


def calculate_psnr(mse):
    return 10 * torch.log10(1. / mse)


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return np.round(t2np(gray))


def generate_diff_data(train_config, test, reference, q_container, images, flags):
    # image dimensions
    dim_h = train_config.dataset_info.h
    dim_w = train_config.dataset_info.w

    # calculate diff of test and reference
    diff_base = torch.abs(test - reference.view(dim_w * dim_h, -1))

    # calculate mse between test and reference
    mse = calculate_mse(diff_base)
    q_container.mse.append(mse.item())

    # calculate peak signal-to-noise ratio
    if "psnr" in flags:
        psnr = calculate_psnr(mse)
        q_container.psnr.append(psnr)

    # calculate ssim
    if "ssim" in flags:
        # NOTE: can be moved to the top if pyrtools available on Windows or other image pyramid library is used
        from util.IW_SSIM_PyTorch import IW_SSIM
        iw_ssim = IW_SSIM(use_cuda=True)

        ssim = iw_ssim.test(rgb2gray(reference.view(dim_w, dim_h, -1)), rgb2gray(test.view(dim_w, dim_h, -1)))
        q_container.ssim.append(ssim)

    diff_base = t2np(diff_base)

    # convert diff to image data
    if not images:
        diff = np.clip(diff_base.reshape(dim_h, dim_w, -1)[:, :, :3], 0., 1.)
        diff_img8bit = (diff[None] * 255).astype(np.uint8)

        if q_container.diff_data is None:
            q_container.diff_data = diff_img8bit
        else:
            q_container.diff_data = np.concatenate((q_container.diff_data, diff_img8bit), axis=0)
    else:
        if q_container.diff_data is None:
            q_container.diff_data = []
        q_container.diff_data.append(diff_base)

    if not images:
        square_diff = np.clip(np.multiply(diff_base, diff_base).reshape(dim_h, dim_w, -1)[:, :, :3], 0., 1.)
        square_diff_img8bit = (square_diff[None] * 255).astype(np.uint8)

        if q_container.square_diff_data is None:
            q_container.square_diff_data = square_diff_img8bit
        else:
            q_container.square_diff_data = np.concatenate((q_container.square_diff_data, square_diff_img8bit), axis=0)
    else:
        if q_container.square_diff_data is None:
            q_container.square_diff_data = []
        q_container.square_diff_data.append(np.multiply(diff_base, diff_base))


def generate_flip_data(train_config, test, reference, q_container, images):
    flip = FLIP()

    # flip standard values
    monitor_distance = 0.7
    monitor_width = 0.7
    monitor_resolution_x = 3840
    pixels_per_degree = monitor_distance * (monitor_resolution_x / monitor_width) * (np.pi / 180)

    # image dimensions
    dim_h = train_config.dataset_info.h
    dim_w = train_config.dataset_info.w

    # color map
    magma = cm.get_cmap('magma')

    # reshape test and reference to flip expectation NxCxHxW
    test = test.view((1, dim_h, dim_w, -1))
    reference = reference.view((1, dim_h, dim_w, -1))

    test_new = test.permute(0, 3, 1, 2)
    reference_new = reference.permute(0, 3, 1, 2)

    # compute flip and convert flip values to color map values
    flip_loss = flip.compute_flip(test_new, reference_new, pixels_per_degree)
    q_container.flip.append(torch.mean(flip_loss).item())

    flip_loss_transformed = magma(flip_loss.detach().cpu().numpy().reshape(dim_h, dim_w, 1))

    # convert flip image
    if not images:
        flip_loss_transformed = np.clip(flip_loss_transformed.reshape(dim_h, dim_w, -1)[:, :, :3], 0., 1.)
        flip_img8bit = (flip_loss_transformed[None] * 255).astype(np.uint8)

        if q_container.flip_data is None:
            q_container.flip_data = flip_img8bit
        else:
            q_container.flip_data = np.concatenate((q_container.flip_data, flip_img8bit), axis=0)
    else:
        if q_container.flip_data is None:
            q_container.flip_data = []
        q_container.flip_data.append(flip_loss_transformed[:, :, :, :3])


def generate_data(train_config, flags, reference_video=None):
    count_flops = False
    image_macs = []
    image_macs_pp = []

    dim_h = train_config.dataset_info.h
    dim_w = train_config.dataset_info.w

    if reference_video is not None:
        # scale image data if config has scale != 1
        if train_config.dataset_info.scale != 1:
            video_scaled = np.zeros((len(reference_video), dim_h, dim_w, 3))
            for i, img in enumerate(reference_video):
                video_scaled[i] = cv2.resize(img, (dim_h, dim_w), interpolation=cv2.INTER_AREA)

            reference_video = video_scaled

        train_config.store_camera_options()
        train_config.config_file.camPath = "cam_path"
        train_config.config_file.camType = "PredefinedCamera"
        train_config.config_file.videoFrames = -1

        dataset = CameraViewCellDataset(train_config.config_file, train_config, train_config.dataset_info)

        desc = "Generating diff and flip videos"
        suffix = "video"
    else:
        os.makedirs(f"{train_config.outDir}/eval/", exist_ok=True)

        if "complexity" in flags:
            count_flops = True
            for i in range(len(train_config.models)):
                train_config.models[i] = flops_counter.add_flops_counting_methods(train_config.models[i])

        dataset = train_config.test_dataset

        desc = "Generating diff and flip images"
        suffix = "images"

    q_cont = QualityContainer()

    # generate test image here to reduce time
    for i in tqdm(range(len(dataset)), desc=desc, position=0, leave=True):
        # acquire test and reference data
        img_data = create_sample_wrapper(dataset[i], train_config, True)

        if count_flops:
            for k in range(len(train_config.models)):
                train_config.models[k].start_flops_count(ost=None, verbose=False, ignore_list=[])

        test = torch.zeros((dim_h * dim_w, 3), device=train_config.device, dtype=torch.float32)
        start_index = 0
        for batch in img_data.batches(train_config.config_file.inferenceChunkSize):
            img_part, _ = train_config.inference(batch, gradient=False, is_inference=True)

            end_index = min(start_index + train_config.config_file.inferenceChunkSize, dim_w * dim_h)

            test[start_index:end_index, :3] = img_part[-1][:train_config.config_file.inferenceChunkSize, :3]
            start_index = end_index

        if count_flops:
            total_macs = 0
            for k in range(len(train_config.models)):
                macs, params = train_config.models[k].compute_average_flops_cost()
                total_macs += macs
                train_config.models[k].stop_flops_count()

            image_macs.append(total_macs * train_config.dataset_info.w * train_config.dataset_info.h)
            image_macs_pp.append(total_macs)

        # just so we can save the created images for debug purposes
        if flags is not None and "output_images" in flags:
            test_clone = torch.clone(test)
            test_clone = t2np(test_clone)

            test_clone = np.clip(test_clone.reshape(train_config.dataset_info.h,
                                                    train_config.dataset_info.w, -1), 0., 1.)[None]

            if q_cont.out_data is None:
                q_cont.out_data = test_clone
            else:
                q_cont.out_data = np.concatenate((q_cont.out_data, test_clone), axis=0)
            del test_clone

        if reference_video is None:
            reference = img_data.get_train_target(-1)
        else:
            reference = (reference_video[i]).astype(np.float32)
            reference = reference / 255
            reference = torch.from_numpy(reference).to(train_config.device)

        # generate data in separate functions to get rid of intermediate values
        generate_diff_data(train_config, test, reference, q_cont, reference_video is None, flags)
        if "flip" in flags:
            generate_flip_data(train_config, test, reference, q_cont, reference_video is None)

    # save data to disk
    if reference_video is not None:
        train_config.restore_camera_options()

        print("Saving diff and flip videos")
        imageio.mimwrite(os.path.join(train_config.outDir, f"_diff.mp4"), q_cont.diff_data, fps=30, quality=8)
        imageio.mimwrite(os.path.join(train_config.outDir, f"_square_diff.mp4"), q_cont.square_diff_data, fps=30, quality=8)
        if "flip" in flags:
            imageio.mimwrite(os.path.join(train_config.outDir, f"_flip.mp4"), q_cont.flip_data, fps=30, quality=8)
        if q_cont.out_data is not None:
            imageio.mimwrite(os.path.join(train_config.outDir, f"_out.mp4"), q_cont.out_data, fps=30, quality=8)
    else:
        for i in tqdm(range(len(q_cont.diff_data)), desc="Saving diff and flip images", position=0, leave=True):
            save_img(q_cont.diff_data[i], train_config.dataset_info, f"{train_config.outDir}/eval/{i}_diff_{q_cont.diff_data[i].mean()}.png")
            save_img(q_cont.square_diff_data[i], train_config.dataset_info, f"{train_config.outDir}/eval/{i}_square_diff_{q_cont.square_diff_data[i].mean()}.png")
            if "flip" in flags:
                save_img(q_cont.flip_data[i], train_config.dataset_info, f"{train_config.outDir}/eval/{i}_flip_{q_cont.flip_data[i].mean()}.png")
            if q_cont.out_data is not None:
                save_img(q_cont.out_data[i], train_config.dataset_info, f"{train_config.outDir}/eval/{i}_out.png")

    if count_flops:
        with open(os.path.join(train_config.outDir, "complexity.txt"), "w") as f:
            cma_macs = 0
            cma_macs_pp = 0

            for idx in range(len(image_macs)):
                macs = image_macs[idx]
                macs_pp = image_macs_pp[idx]

                f.write(f"{idx} - {macs} - {macs_pp}\n")

                # to possibly avoid overflows, we calculate the cumulative moving averages only
                cma_macs = cma_macs + (macs - cma_macs) / (idx + 1)
                cma_macs_pp = cma_macs_pp + (macs_pp - cma_macs_pp) / (idx + 1)

            f.write(f"{cma_macs} : {cma_macs_pp}\n")

    # write quality info to .txt and .csv file
    with open(f"{train_config.outDir}/image_quality_{suffix}.txt", "w") as f:
        for idx, mse in enumerate(q_cont.mse):
            f.write(f"image={idx} mse={mse:.4f} psnr="
                    f"{q_cont.psnr[idx] if 'psnr' in flags else -1.:.4f} "
                    f"ssim="
                    f"{q_cont.ssim[idx] if 'ssim' in flags else -1.:.4f} "
                    f"flip_loss="
                    f"{q_cont.flip[idx] if 'flip' in flags else -1.:.4f}\r")

    with open(f"{train_config.outDir}/image_quality_{suffix}.csv", "w") as c:
        c.write(f"mse,psnr,ssim,flip\r")
        for idx, mse in enumerate(q_cont.mse):
            c.write(f"{mse},{q_cont.psnr[idx] if 'psnr' in flags else -1.},"
                    f"{q_cont.ssim[idx] if 'ssim' in flags else -1.},"
                    f"{q_cont.flip[idx] if 'flip' in flags else -1.}\r")


def load_reference_video(data_path):
    # look for folder called 'reference_video'
    ref_path = os.path.join(data_path, "reference_video")

    if os.path.exists(ref_path):
        reference_video = []

        for i in tqdm(range(360), desc=f'Loading reference video data', position=0, leave=True):
            rgba_image = Image.open(os.path.join(ref_path, f"{i}.png"))
            rgb_image = rgba_image.convert('RGB')
            reference_video.append(np.array(rgb_image))
    else:
        print(f"Warning: no directory named {ref_path} found! adding 'videos' to skip list")
        return None

    return reference_video


def get_network_size(train_config):
    print("Getting network size", end="")

    # get network size
    total_params = 0
    line_buffer = []
    for model in train_config.models:
        for name, params in model.named_parameters():
            if params.requires_grad:
                num_params = np.prod(params.size())
                if params.dim() > 1:
                    line = f"{num_params} = {'x'.join(str(x) for x in list(params.size()))} ({name})"
                else:
                    line = f"{num_params} ({name})"

                line_buffer.append(line)
                total_params += num_params

    line_buffer.insert(0, f"{total_params} total params")
    with open(os.path.join(train_config.outDir, "network_description.txt"), "w") as f:
        for line in line_buffer:
            f.write(f"{line}\n")
    print(" - DONE!")


def evaluate(train_config, reference_video, evaluations):
    if not hasattr(train_config, 'outDir'):
        train_config.outDir = train_config.logDir

    if "opt" in evaluations and not train_config.config_file.trainWithGTDepth:
        print(f"Rendering _opt.mp4")
        train_config.store_camera_options()
        train_config.config_file.camPath = "cam_path"
        train_config.config_file.camType = "PredefinedCamera"
        train_config.config_file.videoFrames = -1
        render_video(train_config, vid_name="_opt", out_dir=train_config.outDir)
        train_config.restore_camera_options()

    # get network size
    if "complexity" in evaluations:
        get_network_size(train_config)

    with torch.cuda.device(train_config.device):
        torch.cuda.empty_cache()

    # get image data
    if "images" in evaluations:
        generate_data(train_config, evaluations)

    with torch.cuda.device(train_config.device):
        torch.cuda.empty_cache()

    # get video data
    if "videos" in evaluations and not train_config.config_file.trainWithGTDepth:
        generate_data(train_config, evaluations, reference_video)

    if "output_videos" in evaluations and not train_config.config_file.trainWithGTDepth:
        # Overwrite the "video" names in the train_config with the requested video names
        if train_config.evaluation_cam_path is not None and len(train_config.evaluation_cam_path) > 0:
            for cam_path in train_config.evaluation_cam_path:
                print(f"Rendering output video {cam_path}")
                train_config.store_camera_options()
                train_config.config_file.camPath = cam_path
                train_config.config_file.camType = "PredefinedCamera"
                train_config.config_file.videoFrames = -1
                render_video(train_config, vid_name=cam_path, out_dir=train_config.outDir)
                train_config.restore_camera_options()
        else:
            print("Warning: output_videos was supplied for evaluation but no camera path (--camPath) was supplied!")

    if "export" in evaluations:
        export_onnx(train_config=train_config, out_dir=os.path.join(train_config.outDir, 'exported_model'))

    # copy opt.txt to eval folder to signify completed evaluation
    if os.path.exists(os.path.join(train_config.logDir, "opt.txt")):
        copyfile(os.path.join(train_config.logDir, "opt.txt"), os.path.join(train_config.outDir, "eval", "opt.txt"))


def get_optimal_epoch(path):
    with open(os.path.join(path, "opt.txt"), "r") as f:
        line = f.readline()
        match = re.search(r'\d+$', line)
        epoch = line[match.start():match.end()]

    return epoch


def load_config(data_path, device_id, path, evaluations, skip, cl_out_dir=None, skip_if_already_done_once=True):
    c_file = os.path.join(path, "config.ini")

    experiment = ""
    orig_path = os.path.join(path, '')

    if path.endswith(f"-D") or path.endswith(f"-D{os.path.sep}"):
        print(f"diff and flip of depth not yet supported")
        return 1, None

    # strip last 2 folders and possible trailing / from log path
    ctr = 0
    while ctr < 2:
        path, tail = os.path.split(path)
        if not tail == '':
            ctr += 1
            if ctr == 1:
                experiment = tail

    print(f"Evaluating {experiment}")

    if not os.path.exists(c_file):
        print(f"No config.ini found!")
        return 1, None

    # get current optimal epoch
    try:
        optimal_epoch = get_optimal_epoch(orig_path)
    except FileNotFoundError:
        print(f"No optimal epoch found - using latest weights")
        optimal_epoch = None

    # add evaluations to the list if none have been passed to the script
    if len(evaluations) == 0:
        opt_filenames = ["_opt.mp4", "_opt_0.mp4", "_opt_1.mp4"]
        opt_files_exist = False

        for filename in opt_filenames:
            opt_files_exist = opt_files_exist or os.path.exists(os.path.join(orig_path, filename))

        if not opt_files_exist and "opt" not in skip:
            evaluations.append("opt")

        eval_categories = ["complexity", "images", "videos", "output_images", "output_videos", "export"]

        for e in eval_categories:
            if e not in skip:
                evaluations.append(e)

    try:
        config = Config.init(c_file, only_known_args=True)
        if isinstance(config, configargparse.ArgParser):
            config, unknown = config.parse_known_args(['-c', c_file])
    except SystemExit:
        print(f"Errors in config file!")
        return 1, None

    # replace paths with command line values
    config.data = data_path
    config.logDir = path

    dataset_name = os.path.basename(os.path.normpath(config.data))
    experiment_name = os.path.basename(os.path.normpath(orig_path))
    out_dir = orig_path
    if cl_out_dir is not None:
        out_dir = os.path.join(cl_out_dir, dataset_name, experiment_name)

    os.makedirs(out_dir, exist_ok=True)

    # look for previous eval
    try:
        evaluated_epoch = get_optimal_epoch(os.path.join(out_dir, "eval"))
    except FileNotFoundError:
        print(f"No previous evaluation found - continuing")
        evaluated_epoch = None

    if evaluated_epoch is not None and optimal_epoch is not None:
        if optimal_epoch == evaluated_epoch and (len(evaluations) == 0 or skip_if_already_done_once):
            print(f"Evaluation already performed for this optimal epoch!")
            return 2, None

    # replace device id in config with command line device id
    config.device = device_id

    # fallback for missing lossWeights
    while len(config.lossWeights) < len(config.losses):
        config.lossWeights.append(1)

    # initialize config and load optimal or latest weights
    train_config = TrainConfig()
    train_config.initialize(config, log_path=orig_path, training=False)

    if cl_out_dir is not None:
        train_config.outDir = out_dir
    else:
        train_config.outDir = train_config.logDir

    if train_config.f_out[-1].n_feat != 3 and train_config.f_out[-1].n_feat != 4:
        print(f"Output features not 3 or 4!")
        return 1, None

    checkpoints = [os.path.join(config.logDir, f) for f in sorted(os.listdir(os.path.join(config.logDir))) if
                   config.checkPointName in f]

    if len(checkpoints) == 0:
        train_config.load_latest_weights()
    else:
        train_config.load_specific_weights(config.checkPointName)

    return 0, train_config


def main():
    p = configargparse.ArgParser()
    p.add_argument('-d', '--device', default=0, type=int, help="cuda device to perform the eval on")
    p.add_argument('-data', '--data', required=True, type=str)
    p.add_argument('-log', '--logDir', default=[], action='append', type=str,
                   help="list containing the nets to be evaluated. if the first does not contain 'config.ini',"
                        " all subdirectories will be added as paths")
    p.add_argument('-e', '--eval', default=[], action='append', type=str,
                   choices=["complexity", "opt", "images", "videos", "output_images", "output_videos", "debug", "export"],
                   help="parameters to recreate the given results, even if evaluation already performed")
    p.add_argument('-s', '--skip', default=[], action='append', type=str,
                   choices=["complexity", "opt", "images", "videos", "output_images", "output_videos", "flip", "psnr", "ssim", "export"],
                   help="evaluations to skip")
    p.add_argument('-o', '--outDir', default=None, type=str,
                   help='output base directory. default - same directory as the input logDir.')
    p.add_argument('--skipIfAlreadyDone', default=True, action="store_false",
                   help="whether or not to skip the full evaluation if we ever did one before (contains opt.txt)")
    p.add_argument('--camPath', default=[], action='append', type=str, help="cam path that is used for output_videos")
    p.add_argument('--inferenceChunkSize', default=4096, type=int)
    cl = p.parse_args()

    device_id = cl.device
    data_path = cl.data
    skip = cl.skip

    data_set_path, data_set = os.path.split(data_path)
    if data_set == '':
        _, data_set = os.path.split(data_set_path)

    if not os.path.exists(os.path.join(cl.logDir[0], "config.ini")):
        paths = []
        for subdir in sorted(os.listdir(cl.logDir[0])):
            path = os.path.join(cl.logDir[0], subdir)
            if os.path.isdir(path):
                paths.append(path)
    else:
        paths = cl.logDir

    reference_video = load_reference_video(data_path)
    if reference_video is None:
        skip.append("videos")

    skipped = 0
    prev_done = 0
    eval_performed = 0

    # NOTE: can be removed if pyrtools available on Windows or other image pyramid library is used
    if platform.system() == 'Linux':
        metrics = ["flip", "psnr", "ssim"]
    else:
        metrics = ["flip", "psnr"]
        print(f"Warning: complexity calculation can lead to overflows if not done on Linux!")

    for idx, path in enumerate(paths):
        if len(paths) > 1:
            print(f"{idx+1} of {len(paths)}")

        evaluations = [e for e in cl.eval]
        state, train_config = load_config(data_path, device_id, path, evaluations, skip, cl_out_dir=cl.outDir,
                                          skip_if_already_done_once=cl.skipIfAlreadyDone)

        if train_config is None:
            print(f"Skipping {path}...")

            if state == 1:
                skipped += 1
            elif state == 2:
                prev_done += 1
        else:
            # we look for the data_set name in path and print warning if not found
            if path.find(data_set) == -1:
                print(f"Warning: did not find dataset name {data_set} in path {path}!")

            for m in metrics:
                if m not in skip:
                    evaluations.append(m)

            if cl.inferenceChunkSize is not None:
                train_config.config_file.inferenceChunkSize = cl.inferenceChunkSize

            train_config.evaluation_cam_path = cl.camPath

            evaluate(train_config, reference_video, evaluations)

            eval_performed += 1

            with torch.cuda.device(train_config.device):
                torch.cuda.empty_cache()

            # free allocated memory to be able to run all tests successively
            del train_config
            gc.collect()

    if len(paths) > 1:
        print(f"Performed evaluation on {eval_performed} of {len(paths)} folders")
        print(f"\tSkipped {skipped} folders because of loading issues")
        print(f"\tPrevious completed evaluation found on {prev_done} folders")


if __name__ == '__main__':
    main()
