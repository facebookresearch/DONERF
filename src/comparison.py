# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import statistics
import configargparse

from util.config import Config
from abc import ABC, abstractmethod


class ExperimentResults:
    def __init__(self, path):
        self.config = None
        self.params = -1
        self.gmacs = 0
        self.gmacs_per_pixel = 0
        self.flops = 0
        self.flops_per_pixel = 0
        self.epoch = -1
        self.completed = False
        _, self.experiment_name = os.path.split(path)

        self.ims = None
        self.vds = None

        network_path = os.path.join(path, "network_description.txt")
        complexity_path = os.path.join(path, "complexity.txt")
        img_quality_path = os.path.join(path, "image_quality_images.csv")
        vid_quality_path = os.path.join(path, "image_quality_video.csv")

        params_ok = self.read_network_parameters(network_path)
        complexity_ok = self.read_complexity_info(complexity_path)
        img_quality_ok = os.path.exists(img_quality_path)

        if img_quality_ok:
            self.ims = read_quality_info(img_quality_path)
        if os.path.exists(vid_quality_path):
            self.vds = read_quality_info(vid_quality_path)

        self.completed = params_ok and complexity_ok and img_quality_ok

    def load_config_file(self, path):
        if not os.path.exists(path):
            return

        self.config = Config.init(path)

    def read_network_parameters(self, path):
        if not os.path.exists(path):
            return False

        with open(path, "r") as f:
            line = f.readline()
            match = re.search(r'\d+', line)
            self.params = line[match.start():match.end()]
        return True

    def read_complexity_info(self, path):
        if not os.path.exists(path):
            return False

        with open(path, "r") as f:
            # we are only interested in the last line (average)
            for line in f:
                pass

        if re.search(r'\d+\.\d+ : \d+\.\d+', line) is not None:
            matches = re.findall(r'\d+\.\d+', line)
            self.gmacs = float(matches[0])
            self.gmacs_per_pixel = float(matches[1])
            self.flops = float(self.gmacs) * 2
            self.flops_per_pixel = float(self.gmacs_per_pixel) * 2

        return True

    def read_optimal_epoch(self, path):
        if not os.path.exists(path):
            return

        with open(path, "r") as f:
            line = f.readline()
            match = re.search(r'\d+$', line)
            epoch = line[match.start():match.end()]
        self.epoch = epoch


def read_quality_info(path):
    metric_names = ["mse", "psnr", "ssim", "flip"]
    metrics = {"mse": [], "psnr": [], "ssim": [], "flip": []}

    with open(path, "r") as f:
        for idx, line in enumerate(f):
            if idx > 0:
                matches = re.findall(r'\d+\.\d+', line)
                for i, metric in enumerate(matches):
                    metrics[metric_names[i]].append(float(matches[i]))

    metric_stats = {}
    for metric in metric_names:
        if len(metrics[metric]) > 0:
            stats = [statistics.mean(metrics[metric]), min(metrics[metric]), max(metrics[metric])]
        else:
            stats = [-1, -1, -1]
        metric_stats[metric] = stats

    return metric_stats


# printer base class for different file types
class ComparisonPrinter(ABC):
    file_type = None
    metric_names = ["mse", "psnr", "ssim", "flip"]

    def get_header_line(self):
        return ""

    def get_item_header_line(self):
        return ""

    def get_item_footer_line(self):
        return ""

    def get_footer_line(self):
        return ""

    @abstractmethod
    def result_to_string(self, result):
        pass


class CSVComparisonPrinter(ComparisonPrinter):
    def __init__(self):
        self.file_type = 'csv'

    def get_header_line(self):
        return f"Experiment_Name,Num_Parameters,Epoch,FLOPS,FLOPS_Per_Pixel,MSE_Average,PSNR_Average,SSIM_Average," \
               f"FLIP_Average,MSE_Average_Video,PSNR_Average_Video,SSIM_Average_Video,FLIP_Average_Video\r"

    def result_to_string(self, result):
        r_string = f"{result.experiment_name},{result.params},{result.epoch},{result.flops},{result.flops_per_pixel}"

        for m in self.metric_names:
            r_string += f",{result.ims[m][0]}"

        for m in self.metric_names:
            if result.vds is not None:
                r_string += f",{result.vds[m][0]}"
            else:
                r_string += ",-1"

        return r_string + "\r"


class XMLComparisonPrinter(ComparisonPrinter):
    def __init__(self):
        self.file_type = 'xml'

    def get_header_line(self):
        return '<?xml version="1.0" encoding="UTF-8"?>\r<experiments>\r'

    def get_item_header_line(self):
        return "\t<experiment>\r"

    def get_item_footer_line(self):
        return "\t</experiment>\r"

    def get_footer_line(self):
        return "</experiments>\r"

    def result_to_string(self, result):
        r_string = f"\t\t<name>{result.experiment_name}</name>\r" \
                   f"\t\t<parameters>{result.params}</parameters>\r" \
                   f"\t\t<epoch>{result.epoch}</epoch>\r" \
                   f"\t\t<flops>{result.flops}</flops>\r" \
                   f"\t\t<flops-per-pixel>{result.flops_per_pixel}</flops-per-pixel>\r"

        for m in self.metric_names:
            r_string += f"\t\t<{m}-average>{result.ims[m][0]}</{m}-average>\r"

        for m in self.metric_names:
            if result.vds is not None:
                r_string += f"\t\t<{m}-average-video>{result.vds[m][0]}</{m}-average-video>\r"
            else:
                r_string += f"\t\t<{m}-average-video>-1</{m}-average-video>\r"

        return r_string


def main():
    p = configargparse.ArgParser()
    p.add_argument('-d', '--directory', required=True, type=str, help='path to directory with results')
    p.add_argument('-f', '--format', default='csv', type=str, choices=["csv", "xml"])
    cl = p.parse_args()

    directory = cl.directory

    # add all subdirectories to the list
    paths = []
    for subdir in sorted(os.listdir(directory)):
        path = os.path.join(directory, subdir)
        if os.path.isdir(path):
            paths.append(path)

    # read in all results
    results = []
    for path in paths:
        result = ExperimentResults(path)
        if result.completed:
            results.append(result)

    printer = None
    if cl.format == 'csv':
        printer = CSVComparisonPrinter()
    elif cl.format == 'xml':
        printer = XMLComparisonPrinter()

    # save results to appropriate format
    with open(os.path.join(directory, f"comparison.{printer.file_type}"), "w") as f:
        header = printer.get_header_line()
        if not header == "":
            f.write(header)

        for idx, result in enumerate(results):
            item_header = printer.get_item_header_line()
            if not item_header == "":
                f.write(item_header)

            f.write(f"{printer.result_to_string(result)}")

            item_footer = printer.get_item_footer_line()
            if not item_footer == "":
                f.write(item_footer)

        footer = printer.get_footer_line()
        if not footer == "":
            f.write(footer)


if __name__ == '__main__':
    main()
