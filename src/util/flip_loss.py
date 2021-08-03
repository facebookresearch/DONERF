# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

#########################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#########################################################################

# FLIP: A Difference Evaluator for Alternating Images
# High Performance Graphics, 2020.
# by Pontus Andersson, Jim Nilsson, Tomas Akenine-Moller, Magnus Oskarsson, Kalle Astrom, and Mark D. Fairchild
#
# Pointer to our paper: https://research.nvidia.com/publication/2020-07_FLIP
# code by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller

import torch
import torch.nn.functional as F
import numpy as np


class FLIPLoss():
    def __init__(self):
        self.model = FLIP()

    def __call__(self, outputs, targets):
        value = self.model.forward(outputs, targets)
        return value


class FLIP(torch.nn.Module):
    def __init__(self):
        self.monitor_distance = 0.7
        self.monitor_width = 0.7
        self.monitor_resolution_x = 3840
        self.pixels_per_degree = self.monitor_distance * (self.monitor_resolution_x / self.monitor_width) * (np.pi / 180)
        self.qc = 0.7
        self.qf = 0.5
        self.pc = 0.4
        self.pt = 0.95

    def compute_flip(self, reference, test, pixels_per_degree):
        # Transform reference and test to opponent color space
        reference = color_space_transform(reference, 'srgb2ycxcz')
        test = color_space_transform(test, 'srgb2ycxcz')

        # --- Color pipeline ---
        # Spatial filtering
        s_a, radius_a = generate_spatial_filter(pixels_per_degree, 'A')
        s_rg, radius_rg = generate_spatial_filter(pixels_per_degree, 'RG')
        s_by, radius_by = generate_spatial_filter(pixels_per_degree, 'BY')
        radius = max(radius_a, radius_rg, radius_by)
        filtered_reference = spatial_filter(reference, s_a, s_rg, s_by, radius)
        filtered_test = spatial_filter(test, s_a, s_rg, s_by, radius)

        # Perceptually Uniform Color Space
        preprocessed_reference = hunt_adjustment(color_space_transform(filtered_reference, 'linrgb2lab'))
        preprocessed_test = hunt_adjustment(color_space_transform(filtered_test, 'linrgb2lab'))

        # Color metric
        deltaE_hyab = hyab(preprocessed_reference, preprocessed_test)
        power_deltaE_hyab = torch.pow(deltaE_hyab, self.qc)
        hunt_adjusted_green = hunt_adjustment(color_space_transform(torch.tensor([[[0.0]], [[1.0]], [[0.0]]]).unsqueeze(0), 'linrgb2lab'))
        hunt_adjusted_blue = hunt_adjustment(color_space_transform(torch.tensor([[[0.0]], [[0.0]], [[1.0]]]).unsqueeze(0), 'linrgb2lab'))
        cmax = torch.pow(hyab(hunt_adjusted_green, hunt_adjusted_blue), self.qc).item()
        deltaE_c = redistribute_errors(power_deltaE_hyab, cmax, self.pc, self.pt)

        # --- Feature pipeline ---
        # Extract and normalize Yy component
        ref_y = (reference[:, 0:1, :, :] + 16) / 116
        test_y = (test[:, 0:1, :, :] + 16) / 116

        # Edge and point detection
        edges_reference = feature_detection(ref_y, pixels_per_degree, 'edge')
        points_reference = feature_detection(ref_y, pixels_per_degree, 'point')
        edges_test = feature_detection(test_y, pixels_per_degree, 'edge')
        points_test = feature_detection(test_y, pixels_per_degree, 'point')

        # Feature metric
        deltaE_f = torch.max(torch.abs(torch.norm(edges_reference, dim=1, keepdim=True) - torch.norm(edges_test, dim=1, keepdim=True)),
                             torch.abs(torch.norm(points_test, dim=1, keepdim=True) - torch.norm(points_reference, dim=1, keepdim=True)))
        deltaE_f = torch.pow(((1 / np.sqrt(2)) * deltaE_f), self.qf)
        deltaE_f = torch.clamp(deltaE_f, 0.0, 1.0)  # clamp added to stabilize training

        # --- Final error ---
        return torch.pow(deltaE_c, 1 - deltaE_f)

    def forward(self, outputs, targets):
        deltaE = self.compute_flip(targets, outputs, self.pixels_per_degree)
        return torch.mean(deltaE)


def generate_spatial_filter(pixels_per_degree, channel):
    a1_A = 1
    b1_A = 0.0047
    a2_A = 0
    b2_A = 1e-5  # avoid division by 0
    a1_rg = 1
    b1_rg = 0.0053
    a2_rg = 0
    b2_rg = 1e-5  # avoid division by 0
    a1_by = 34.1
    b1_by = 0.04
    a2_by = 13.5
    b2_by = 0.025
    if channel == "A":  # Achromatic CSF
        a1 = a1_A
        b1 = b1_A
        a2 = a2_A
        b2 = b2_A
    elif channel == "RG":  # Red-Green CSF
        a1 = a1_rg
        b1 = b1_rg
        a2 = a2_rg
        b2 = b2_rg
    elif channel == "BY":  # Blue-Yellow CSF
        a1 = a1_by
        b1 = b1_by
        a2 = a2_by
        b2 = b2_by

    # Determine evaluation domain
    max_scale_parameter = max([b1_A, b2_A, b1_rg, b2_rg, b1_by, b2_by])
    r = np.ceil(3 * np.sqrt(max_scale_parameter / (2 * np.pi ** 2)) * pixels_per_degree)
    r = int(r)
    deltaX = 1.0 / pixels_per_degree
    x, y = np.meshgrid(range(-r, r + 1), range(-r, r + 1))
    z = (x * deltaX) ** 2 + (y * deltaX) ** 2

    # Generate weights
    g = a1 * np.sqrt(np.pi / b1) * np.exp(-np.pi ** 2 * z / b1) + a2 * np.sqrt(np.pi / b2) * np.exp(-np.pi ** 2 * z / b2)
    g = g / np.sum(g)
    g = torch.Tensor(g).unsqueeze(0).unsqueeze(0).cuda()

    return g, r


def spatial_filter(img, s_a, s_rg, s_by, radius):
    # Filters image img using Contrast Sensitivity Functions.
    # Returns linear RGB

    dim = img.size()
    # Prepare image for convolution
    img_pad = torch.zeros((dim[0], dim[1], dim[2] + 2 * radius, dim[3] + 2 * radius), device='cuda')
    img_pad[:, 0:1, :, :] = F.pad(img[:, 0:1, :, :], (radius, radius, radius, radius), mode='replicate')
    img_pad[:, 1:2, :, :] = F.pad(img[:, 1:2, :, :], (radius, radius, radius, radius), mode='replicate')
    img_pad[:, 2:3, :, :] = F.pad(img[:, 2:3, :, :], (radius, radius, radius, radius), mode='replicate')

    # Apply Gaussian filters
    img_tilde_opponent = torch.zeros((dim[0], dim[1], dim[2], dim[3]), device='cuda')
    img_tilde_opponent[:, 0:1, :, :] = F.conv2d(img_pad[:, 0:1, :, :], s_a.cuda(), padding=0)
    img_tilde_opponent[:, 1:2, :, :] = F.conv2d(img_pad[:, 1:2, :, :], s_rg.cuda(), padding=0)
    img_tilde_opponent[:, 2:3, :, :] = F.conv2d(img_pad[:, 2:3, :, :], s_by.cuda(), padding=0)

    # Transform to linear RGB for clamp
    img_tilde_linear_rgb = color_space_transform(img_tilde_opponent, 'ycxcz2linrgb')

    # Clamp to RGB box
    return torch.clamp(img_tilde_linear_rgb, 0, 1)


def hunt_adjustment(img):
    # Applies Hunt adjustment to L*a*b* image img

    # Extract luminance component
    L = img[:, 0:1, :, :]

    # Apply Hunt adjustment
    img_h = torch.zeros(img.size(), device='cuda')
    img_h[:, 0:1, :, :] = L
    img_h[:, 1:2, :, :] = torch.mul((0.01 * L), img[:, 1:2, :, :])
    img_h[:, 2:3, :, :] = torch.mul((0.01 * L), img[:, 2:3, :, :])

    return img_h


def hyab(reference, test):
    # Computes HyAB distance between L*a*b* images reference and test
    delta = reference - test
    return abs(delta[:, 0:1, :, :]) + torch.norm(delta[:, 1:3, :, :], dim=1, keepdim=True)


def redistribute_errors(power_deltaE_hyab, cmax, pc, pt):
    # Re-map error to 0-1 range. Values between 0 and
    # pccmax are mapped to the range [0, pt],
    # while the rest are mapped to the range (pt, 1]
    deltaE_c = torch.zeros(power_deltaE_hyab.size(), device='cuda')
    pccmax = pc * cmax
    deltaE_c = torch.where(power_deltaE_hyab < pccmax, (pt / pccmax) * power_deltaE_hyab, pt + ((power_deltaE_hyab - pccmax) / (cmax - pccmax)) * (1.0 - pt))

    return deltaE_c


def feature_detection(img_y, pixels_per_degree, feature_type):
    # Finds features of type feature_type in image img based on current PPD

    # Set peak to trough value (2x standard deviations) of human edge
    # detection filter
    w = 0.082

    # Compute filter radius
    sd = 0.5 * w * pixels_per_degree
    radius = int(np.ceil(3 * sd))

    # Compute 2D Gaussian
    [x, y] = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sd * sd))

    if feature_type == 'edge':  # Edge detector
        # Compute partial derivative in x-direction
        Gx = np.multiply(-x, g)
    else:  # Point detector
        # Compute second partial derivative in x-direction
        Gx = np.multiply(x ** 2 / (sd * sd) - 1, g)

    # Normalize positive weights to sum to 1 and negative weights to sum to -1
    negative_weights_sum = -np.sum(Gx[Gx < 0])
    positive_weights_sum = np.sum(Gx[Gx > 0])
    Gx = torch.Tensor(Gx)
    Gx = torch.where(Gx < 0, Gx / negative_weights_sum, Gx / positive_weights_sum)
    Gx = Gx.unsqueeze(0).unsqueeze(0).cuda()

    # Detect features
    featuresX = F.conv2d(F.pad(img_y, (radius, radius, radius, radius), mode='replicate'), Gx, padding=0)
    featuresY = F.conv2d(F.pad(img_y, (radius, radius, radius, radius), mode='replicate'), torch.transpose(Gx, 2, 3), padding=0)
    return torch.cat((featuresX, featuresY), dim=1)


def color_space_transform(input_color, fromSpace2toSpace):
    dim = input_color.size()

    if fromSpace2toSpace == "srgb2linrgb":
        input_color = torch.clamp(input_color, 0.0, 1.0)  # clamp added to stabilize training
        limit = 0.04045
        transformed_color = torch.where(input_color > limit, torch.pow((input_color + 0.055) / 1.055, 2.4), input_color / 12.92)

    elif fromSpace2toSpace == "linrgb2srgb":
        input_color = torch.clamp(input_color, 0.0, 1.0)  # clamp added to stabilize training
        limit = 0.0031308
        transformed_color = torch.where(input_color > limit, 1.055 * (input_color ** (1.0 / 2.4)) - 0.055, 12.92 * input_color)

    elif fromSpace2toSpace == "linrgb2xyz" or fromSpace2toSpace == "xyz2linrgb":
        # Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
        # Assumes D65 standard illuminant
        a11 = 10135552 / 24577794
        a12 = 8788810 / 24577794
        a13 = 4435075 / 24577794
        a21 = 2613072 / 12288897
        a22 = 8788810 / 12288897
        a23 = 887015 / 12288897
        a31 = 1425312 / 73733382
        a32 = 8788810 / 73733382
        a33 = 70074185 / 73733382
        A = torch.Tensor([[a11, a12, a13],
                          [a21, a22, a23],
                          [a31, a32, a33]])

        input_color = input_color.view(dim[0], dim[1], dim[2] * dim[3]).cuda()  # NC(HW)
        if fromSpace2toSpace == "xyz2linrgb":
            A = torch.inverse(A)
        transformed_color = torch.matmul(A.cuda(), input_color)
        transformed_color = transformed_color.view(dim[0], dim[1], dim[2], dim[3])

    elif fromSpace2toSpace == "xyz2ycxcz":
        reference_illuminant = color_space_transform(torch.ones(dim), 'linrgb2xyz')
        input_color = torch.div(input_color, reference_illuminant)
        y = 116 * input_color[:, 1:2, :, :] - 16
        cx = 500 * (input_color[:, 0:1, :, :] - input_color[:, 1:2, :, :])
        cz = 200 * (input_color[:, 1:2, :, :] - input_color[:, 2:3, :, :])
        transformed_color = torch.cat((y, cx, cz), 1)

    elif fromSpace2toSpace == "ycxcz2xyz":
        y = (input_color[:, 0:1, :, :] + 16) / 116
        cx = input_color[:, 1:2, :, :] / 500
        cz = input_color[:, 2:3, :, :] / 200

        x = y + cx
        z = y - cz
        transformed_color = torch.cat((x, y, z), 1)

        reference_illuminant = color_space_transform(torch.ones(dim), 'linrgb2xyz')
        transformed_color = torch.mul(transformed_color, reference_illuminant)

    elif fromSpace2toSpace == "xyz2lab":
        reference_illuminant = color_space_transform(torch.ones(dim), 'linrgb2xyz')
        input_color = torch.div(input_color, reference_illuminant)
        delta = 6 / 29
        limit = 0.00885

        input_color = torch.where(input_color > limit, torch.pow(input_color, 1 / 3), (input_color / (3 * delta * delta)) + (4 / 29))

        l = 116 * input_color[:, 1:2, :, :] - 16
        a = 500 * (input_color[:, 0:1, :, :] - input_color[:, 1:2, :, :])
        b = 200 * (input_color[:, 1:2, :, :] - input_color[:, 2:3, :, :])

        transformed_color = torch.cat((l, a, b), 1)

    elif fromSpace2toSpace == "lab2xyz":
        y = (input_color[:, 0:1, :, :] + 16) / 116
        a = input_color[:, 1:2, :, :] / 500
        b = input_color[:, 2:3, :, :] / 200

        x = y + a
        z = y - b

        xyz = torch.cat((x, y, z), 1)
        delta = 6 / 29
        xyz = torch.where(xyz > delta, xyz ** 3, 3 * delta ** 2 * (xyz - 4 / 29))

        reference_illuminant = color_space_transform(torch.ones(dim), 'linrgb2xyz')
        transformed_color = torch.mul(xyz, reference_illuminant)

    elif fromSpace2toSpace == "srgb2xyz":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
    elif fromSpace2toSpace == "srgb2ycxcz":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
    elif fromSpace2toSpace == "linrgb2ycxcz":
        transformed_color = color_space_transform(input_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
    elif fromSpace2toSpace == "srgb2lab":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    elif fromSpace2toSpace == "linrgb2lab":
        transformed_color = color_space_transform(input_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    elif fromSpace2toSpace == "ycxcz2linrgb":
        transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
    elif fromSpace2toSpace == "lab2srgb":
        transformed_color = color_space_transform(input_color, 'lab2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2srgb')
    elif fromSpace2toSpace == "ycxcz2lab":
        transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    else:
        print('The color transform is not defined!')
        transformed_color = input_color

    return transformed_color
