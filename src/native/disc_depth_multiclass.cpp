// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the CC-BY-NC license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>

#include <iostream>
#include <vector>

void cuda_func();
void fill_depth_cuda(
  torch::Tensor features,
  torch::Tensor pixel_ids,
  torch::Tensor img_ids,
  torch::Tensor x,
  int window_size,
  int height,
  int width,
  int num_samples,
  int num_images,
  int center_id,
  int num_features,
  float maxdist,
  float step,
  float depthcutoff,
  int version);

void fill_disc_depth(torch::Tensor features, torch::Tensor pixel_ids, torch::Tensor img_ids, torch::Tensor x,
                     int window_size, int height, int width, int num_samples, int num_images, int center_id,
                     int num_features, double depthcutoff, int version)
{
    //std::cout << "\n extension got called!\n";
    //cuda_func();

    float maxdist = (floor(window_size * 0.5f) + 1) * sqrt(2.0f);
    const float step = 1.0f / num_features;

    fill_depth_cuda(features, pixel_ids, img_ids, x, window_size, height, width, num_samples, num_images, center_id,
                    num_features, maxdist, step, depthcutoff, version);
}


// torch::Tensor d_sigmoid(torch::Tensor z)
// {
//     std::cout << "got called!\n";
//     auto s = torch::sigmoid(z);
//     return (1 - s) * s;
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fill_disc_depth", &fill_disc_depth, "docu");
}
