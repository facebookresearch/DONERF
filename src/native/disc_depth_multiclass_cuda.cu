// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the CC-BY-NC license found in the
// LICENSE file in the root directory of this source tree.


#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <iostream>
#include <stdio.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void cuda_func()
{
    std::cout << "hi from the cu file\n";
}

// workaround for having float atomicMax
// found at https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void kernel_1_t_per_sample(
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> features,
        torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> pixel_ids,
        torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> img_ids,
        torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> x_tensor,
        const int window_size,
        const int height,
        const int width,
        const int num_samples,
        const int num_images,
        const int center_id,
        const int num_features,
        const float maxdist,
        const float depth_cutoff,
        const float step)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    // only last block should have inactive threads
    if (index < num_samples * num_images)
    {
        const int img_id = img_ids[index / num_samples];
        const int pixel_id = pixel_ids[index];

        const int pixel_x = pixel_id % width;
        const int pixel_y = pixel_id / width;

        for (int i = 0; i < window_size; ++i)
        {
            for(int j = 0; j < window_size; ++j)
            {
                const float weight = (1.0f - sqrtf((i - center_id) * (i - center_id) + (j - center_id) * (j - center_id)) / maxdist);
                const int x = max(0, min(width - 1, pixel_x - center_id + i));
                const int y = max(0, min(height - 1, pixel_y - center_id + j));

                float v = x_tensor[img_id][y][x][0];
                int disc = v / step;
                if (disc >= 0 && v < depth_cutoff)
                {
                    disc = min(disc, num_features-1);
                    atomicMax(&features[index][disc], weight);
                }
            }
        }
    }
}

__global__ void kernel_5_t_per_sample(
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> features,
        torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> pixel_ids,
        torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> img_ids,
        torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> x_tensor,
        const int window_size,
        const int height,
        const int width,
        const int num_samples,
        const int num_images,
        const int center_id,
        const int num_features,
        const float maxdist,
        const float depth_cutoff,
        const float step)
{
    const int index = blockIdx.x * blockDim.y + threadIdx.y;

    // only last block should have inactive threads
    if (index < num_samples * num_images)
    {
        const int img_id = img_ids[index / num_samples];
        const int pixel_id = pixel_ids[index];

        const int pixel_x = pixel_id % width;
        const int pixel_y = pixel_id / width;

        const int i = threadIdx.x;
        const float x_dist = (i - center_id) * (i - center_id);
        const int x = max(0, min(width - 1, pixel_x - center_id + i));

        for(int j = 0; j < window_size; ++j)
        {
            const float weight = (1.0f - sqrtf(x_dist + (j - center_id) * (j - center_id)) / maxdist);
            const int y = max(0, min(height - 1, pixel_y - center_id + j));

            float v = x_tensor[img_id][y][x][0];
            int disc = v / step;
            if (disc >= 0 && v < depth_cutoff)
            {
                disc = min(disc, num_features-1);
                atomicMax(&features[index][disc], weight);
            }
        }
    }
}

__global__ void kernel_5_t_per_sample_2_inactive(
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> features,
        torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> pixel_ids,
        torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> img_ids,
        torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> x_tensor,
        const int window_size,
        const int height,
        const int width,
        const int num_samples,
        const int num_images,
        const int center_id,
        const int num_features,
        const float maxdist,
        const float depth_cutoff,
        const float step)
{
    int pixel_per_block_x =  blockDim.x / window_size;
    int pixel_in_block_x = threadIdx.x / window_size;
    int index = blockIdx.x * blockDim.y * pixel_per_block_x + threadIdx.y * pixel_per_block_x + pixel_in_block_x;

    // only last block should have inactive threads
    if (index < num_samples * num_images && pixel_in_block_x < pixel_per_block_x)
    {
        const int img_id = img_ids[index / num_samples];
        const int pixel_id = pixel_ids[index];

        const int pixel_x = pixel_id % width;
        const int pixel_y = pixel_id / width;

        const int i = threadIdx.x % window_size;
        const int x_dist = (i - center_id) * (i - center_id);
        const int x = max(0, min(width - 1, pixel_x - center_id + i));

        for(int j = 0; j < window_size; ++j)
        {
            const float weight = (1.0f - sqrtf(x_dist + (j - center_id) * (j - center_id)) / maxdist);
            const int y = max(0, min(height - 1, pixel_y - center_id + j));

            float v = x_tensor[img_id][y][x][0];
            int disc = v / step;
            if (disc >= 0 && v < depth_cutoff)
            {
                disc = min(disc, num_features-1);
                atomicMax(&features[index][disc], weight);
            }
        }
    }
}


void fill_depth_cuda(torch::Tensor features, torch::Tensor pixel_ids, torch::Tensor img_ids, torch::Tensor x,
               const int window_size, const int height, const int width, const int num_samples,
               const int num_images, const int center_id, const int num_features, const float maxdist,
               const float step, float depth_cutoff, const int version)
{
    CHECK_INPUT(features);
    CHECK_INPUT(pixel_ids);
    CHECK_INPUT(img_ids);
    CHECK_INPUT(x);

    cudaError_t err;

    if (version == 0)
    {
        const int threads = 1024;
        int num_blocks = (num_images * num_samples + threads - 1) / threads;

        kernel_1_t_per_sample<<<num_blocks, threads>>>(
            features.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            pixel_ids.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            img_ids.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            x.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            window_size,
            height,
            width,
            num_samples,
            num_images,
            center_id,
            num_features,
            maxdist,
            depth_cutoff,
            step);
    }
    else if (version == 1)
    {
        dim3 block_dim(window_size, 64, 1);
        // we only use y as size of block, because we have window_size threads per sample
        const int block_size = block_dim.y;
        const int num_blocks = (num_images * num_samples + block_size - 1) / block_size;
        dim3 grid_dim(num_blocks, 1, 1);

        kernel_5_t_per_sample<<<grid_dim, block_dim>>>(
            features.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            pixel_ids.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            img_ids.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            x.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            window_size,
            height,
            width,
            num_samples,
            num_images,
            center_id,
            num_features,
            maxdist,
            depth_cutoff,
            step);
    }
    else if (version == 2)
    {
        dim3 block_dim(32, 32, 1);
        // we use y and 32 / window_size as size of block, so x of block is 1 warp
        const int block_size = block_dim.y * (block_dim.x / window_size);
        const int num_blocks = (num_images * num_samples + block_size - 1) / block_size;
        dim3 grid_dim(num_blocks, 1, 1);

        kernel_5_t_per_sample_2_inactive<<<grid_dim, block_dim>>>(
            features.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            pixel_ids.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            img_ids.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            x.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            window_size,
            height,
            width,
            num_samples,
            num_images,
            center_id,
            num_features,
            maxdist,
            depth_cutoff,
            step);
    }

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
