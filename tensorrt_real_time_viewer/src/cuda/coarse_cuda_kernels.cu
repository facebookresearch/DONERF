
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include "device_launch_parameters.h"

#include "../include/cuda/donerf_cuda_kernels.cuh"
#include "../include/cuda/helper_math.h"
#include "../include/cuda/donerf_cuda_helper.h"

#include <cuda.h>
#include <iostream>
#include <assert.h> 


__device__ void encodePosDir(int num_freqs, float* freq_band, float3* features, float3 val, int offset)
{
  float3 v_f;
  for (int c = 0; c < num_freqs; c++)
  {
    v_f = val * freq_band[c];
    features[offset + c * 2 + 0] = make_float3(sin(v_f.x), sin(v_f.y), sin(v_f.z));
    features[offset + c * 2 + 1] = make_float3(cos(v_f.x), cos(v_f.y), cos(v_f.z));
  }
}


template<int t_n_weights, int t_n_samples>
__global__ void samplePDF_U(int size, int batch_size, int batch_offset, 
  float* features, int num_ray_samples, float* z_vals, float* weights, int num_weights, 
  float z_near, float z_far, float min_d_range, float max_d_range)
{
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= batch_size)
    return;

  int idx = id;

  float depth_sum = 0.0f;

  float current_depth = 0.0f;

  int cdf_size = num_weights + 1;
  float cdfs_[t_n_weights +1];
  float depths_[t_n_weights +1];

#pragma unroll
  for (int k = 1; k < t_n_weights+1; k++)
  {
    current_depth = weights[idx * (t_n_weights+2) + k ] + 1e-5f; // depth transform;   weight += 1e-5 // offset   //weights[..., 1:-1] 
    depths_[k-1] = current_depth;
    depth_sum += current_depth; //  torch.sum(weights, -1, keepdim=True)
  }

  float cdf = 0.0f;
  float u = 0;
  float u_step = 1.0f / (t_n_samples  -1);

  int below[t_n_samples +1];
  int above[t_n_samples+1];
  int ind_c = 0;

#pragma unroll
  for (int k = 0; k < (t_n_weights +1); k++) // cdf  63
  {
    float pdf = 0;
    if (k < t_n_weights)
      pdf = depths_[k] / depth_sum;

    while (cdf > u && ind_c < t_n_samples+1)
    {
      below[ind_c] = max(0, k - 1);
      above[ind_c] = min(cdf_size - 1, k);

      ind_c++;
      u += u_step;
    }
    cdfs_[k] = cdf;
    cdf += pdf;
  }

#pragma unroll
  for (int l = ind_c; l < (t_n_samples +1); l++) // 128
  {
    below[l] = max(0, cdf_size - 2);
    above[l] = min(cdf_size - 1, cdf_size - 2);
  }

  u = 0;
  int insert_idx = idx *(t_n_samples + t_n_weights + 2);
  int counter = 0;
  int counterb = 0;

  float u_old = z_near;
  float u_old_c = logtToWorld(z_near, min_d_range, max_d_range);
  float old_step =  (z_far - z_near)/ ((t_n_weights + 1));


#pragma unroll
  for (int l = 0; l < t_n_samples ; l++)
  {
    float cdf_g0 = cdfs_[below[l]];
    float cdf_g1 = cdfs_[above[l]];

    float bin_g0 = getBinMid((t_n_weights + 1), below[l], z_near, z_far, min_d_range, max_d_range); // .5 * (previous_z_vals[..., 1:] + previous_z_vals[..., :-1])
    float bin_g1 = getBinMid((t_n_weights + 1), above[l], z_near, z_far, min_d_range, max_d_range);

    float denom = cdf_g1 - cdf_g0;
    if (denom < 1e-5f)
      denom = 1.0f;

    float t = (u - cdf_g0) / denom;
    float sample = bin_g0 + t * (bin_g1 - bin_g0);


    while (u_old_c < sample && u_old <= z_far)
    {
      z_vals[insert_idx] = u_old_c;
      insert_idx++;
      counter++;

      u_old += old_step;
      u_old_c = logtToWorld(u_old, min_d_range, max_d_range);
    }

    {
      z_vals[insert_idx] = sample;
      counterb++;
      insert_idx++;
    }

    u += u_step;
  }
  while (counter < (t_n_weights + 2))
  {
    z_vals[insert_idx] = logtToWorld(u_old, min_d_range, max_d_range);

    insert_idx++;
    u_old += old_step;
    counter++;
  }
}

// fromposes2 with ray calc
__global__ void RayMarchCoarse(int size, int batch_size, int batch_offset, int additional_samples,
  float* features, float* features_clean, float * rot_mat, float3 p,
  int depth_transform, float z_near, float z_far, float min_d_range, float max_d_range, float* z_vals,
  float3 center, float max_depth,
  int encoding, int num_dir_freqs, int num_pos_freqs, float* dir_freq_bands, float* pos_freq_bands )
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int k = idx % additional_samples;
  idx = idx / additional_samples;
  if (idx >= batch_size || k >= additional_samples)
    return;


  int pic_x = (idx + batch_offset) % size;
  int pic_y = (idx + batch_offset) / size;

  if (pic_y >= size)
    return;

  float3 fc = reinterpret_cast<float3*>(features_clean)[( pic_x + pic_y * size)*2];

  float3 rt1 = reinterpret_cast<float3*>(rot_mat)[0];
  float3 rt2 = reinterpret_cast<float3*>(rot_mat)[1];
  float3 rt3 = reinterpret_cast<float3*>(rot_mat)[2];

  float3 d;
  d.x = fc.x * rt1.x + fc.y * rt1.y + fc.z * rt1.z;
  d.y = fc.x * rt2.x + fc.y * rt2.y + fc.z * rt2.z;
  d.z = fc.x * rt3.x + fc.y * rt3.y + fc.z * rt3.z;

  float3* features_r = reinterpret_cast<float3*>(features);

  d = normalize(d);

  int n_features_r = (1 + num_dir_freqs * 2 + 1 + num_pos_freqs * 2);
  int feature_idx_r = idx * additional_samples * n_features_r;


  float step = 1.0f / (float)(additional_samples-1);

  float t_val = step * k;
  float z_val = z_near * (1 - t_val) + z_far * t_val;

  if (depth_transform)
    z_val = logtToWorld(z_val, min_d_range, max_d_range);
  else
    z_val = lintToWorld(z_val, min_d_range, max_d_range);


  z_vals[idx * additional_samples + k] = z_val;


  float3 p_ = p + d * z_val;

  float3 np = normalizationInverseSqrtDistCentered(p_, center, max_depth);

  int p_start_r = feature_idx_r + k * n_features_r;
  int d_start_r = feature_idx_r + k * n_features_r + 1 + num_pos_freqs * 2;

  features_r[p_start_r] = np;
  features_r[d_start_r] = d;

  if (encoding)
    encodePosDir(num_pos_freqs, pos_freq_bands, features_r, np, p_start_r + 1);

  if (encoding)
    encodePosDir(num_dir_freqs, dir_freq_bands, features_r, d, d_start_r + 1);
}


template<int t_n_last_samples, int t_n_additonal_samples, bool sqrt_norm>
__global__ void RayMarchFromCoarse(int size, int batch_size, int batch_offset,
  float* features, float* features_last, float* features_clean, float z_step,
  float noise_amplitude, int num_ray_samples, int depth_transform, float z_near, float z_far, float min_d_range, float max_d_range, float* z_vals,
  float3 center, float max_depth,
  int encoding, int num_dir_freqs, int num_pos_freqs, float* dir_freq_bands, float* pos_freq_bands,
  float* depth, int num_depths, int use_z_vals, float3 p, float* rot_mat)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int k = idx % (t_n_additonal_samples + t_n_last_samples);
  idx = idx / (t_n_additonal_samples + t_n_last_samples);


  int pic_x = (idx + batch_offset) % size;
  int pic_y = (idx + batch_offset) / size;

  if (idx >= batch_size)
    return;

  float3* features_r = reinterpret_cast<float3*>(features);

  int n_features_r = (1 + num_dir_freqs * 2 + 1 + num_pos_freqs * 2);

  float3 fc = reinterpret_cast<float3*>(features_clean)[(pic_x + pic_y * size) * 2];

  float3 rt1 = reinterpret_cast<float3*>(rot_mat)[0];
  float3 rt2 = reinterpret_cast<float3*>(rot_mat)[1];
  float3 rt3 = reinterpret_cast<float3*>(rot_mat)[2];

  float3 d;
  d.x = fc.x * rt1.x + fc.y * rt1.y + fc.z * rt1.z;
  d.y = fc.x * rt2.x + fc.y * rt2.y + fc.z * rt2.z;
  d.z = fc.x * rt3.x + fc.y * rt3.y + fc.z * rt3.z;
  
  d = normalize(d);   

  int feature_idx_r = idx * (t_n_last_samples + t_n_additonal_samples) * n_features_r;  

  float step = 1.0f / (t_n_last_samples + t_n_additonal_samples);

  float t_val = step * k;
  float z_val = z_near * (1 - t_val) + z_far * t_val;

  z_val = z_vals[idx * (t_n_last_samples + t_n_additonal_samples) + k];

  float3 p_ = p + d * z_val;
  float3 np = p_;
  
  if(sqrt_norm)
    np = normalizationInverseSqrtDistCentered(p_, center, max_depth);

  int p_start_r = feature_idx_r + k * n_features_r;
  int d_start_r = feature_idx_r + k * n_features_r + 1 + num_pos_freqs * 2;


  features_r[p_start_r] = np;
  features_r[d_start_r] = d;

  if (encoding)
    encodePosDir(num_pos_freqs, pos_freq_bands, features_r, np, p_start_r + 1);

  if (encoding)
    encodePosDir(num_dir_freqs, dir_freq_bands, features_r, d, d_start_r + 1);
}


__global__ void nerf_raw_2_output_weights(float* shading_output, float* weights, float* z_vals, float* features, 
  int size, int num_samples, int batch_size, int batch_offset, 
  float z_near, float z_far, float min_d_range, float max_d_range)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size)
    return;

  int x_ = (idx + batch_offset) % size;
  int y_ = (idx + batch_offset) / size;

  if (x_ >= size || y_ >= size)
    return;

  int start = idx * 4 * num_samples;
  int start_2 = idx * num_samples;

  float last_prod = 1.0f;

  for (int k = 0; k < num_samples; k++)//  rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
  {
    // dists = z_vals[..., 1:] - z_vals[..., :-1];    
    // dists = torch.cat([dists, (torch.ones(1, device = raw.device) * 1e10).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    float dist = (k < num_samples - 1)
      ? z_vals[start_2 + k + 1] - z_vals[start_2 + k]
      : 1e10f;

    float alpha = raw2alpha(shading_output[start + 3 + k * 4], dist);

    // weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device = raw.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    float prod = last_prod * (1.0f - alpha + 1e-10f);
    float weight = alpha * last_prod;  

    weights[start_2 + k] = weight;

    last_prod = prod;
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HOST CODE 

void updateRayMarchCoarse(float* features, float* features_clean, float* features_last, 
  float x, float y, float z, float* rotation_matrix, int width, int height,
  int num_dir_enc, int num_pos_enc, int encoding, float* dir_freq_bands, float* pos_freq_bands,
  float z_step, int num_ray_samples, int depth_transform, 
  float z_near, float z_far, float min_d_range, float max_d_range, float* z_vals,
  float vc_center_x, float vc_center_y, float vc_center_z, float max_depth,
  int batch_offset, int batch_size)
{
  cudaDeviceSynchronize();

  dim3 dimBlock2(512);
  dim3 dimGrid2((batch_size * num_ray_samples + dimBlock2.x - 1) / dimBlock2.x);

  RayMarchCoarse<<<dimGrid2, dimBlock2>>>(width, batch_size, batch_offset, num_ray_samples,
    features, features_clean, rotation_matrix, make_float3(x,y,z),
    depth_transform, z_near, z_far, min_d_range, max_d_range, z_vals,
    make_float3(vc_center_x, vc_center_y, vc_center_z), max_depth,
    encoding, num_dir_enc, num_pos_enc, dir_freq_bands, pos_freq_bands);

  CUDA_CHECK;
  cudaDeviceSynchronize();
}

void updateRayMarchFromCoarse(float* features, float* features_clean, float* features_last, 
  float x, float y, float z, float* rotation_matrix, int width, int height,
  int num_dir_enc, int num_pos_enc, int encoding, float* dir_freq_bands, float* pos_freq_bands,
  float z_step, float noise_amplitude, int num_ray_samples, int depth_transform, 
  float z_near, float z_far, float min_d_range, float max_d_range, float* z_vals,
  float vc_center_x, float vc_center_y, float vc_center_z, float max_depth,
  int batch_offset, int batch_size, std::string type,  float* cdf, float* prev_z_vals, float* weights, float* out_prev, int out_prev_size )
{
  dim3 dimBlock4(16);
  dim3 dimGrid4((batch_size + dimBlock4.x - 1) / dimBlock4.x);

  nerf_raw_2_output_weights<<<dimGrid4, dimBlock4>>>(out_prev ,(float*) weights, nullptr, features, 
    width, 64, batch_size, batch_offset, z_near, z_far, min_d_range, max_d_range);

  cudaDeviceSynchronize();
  CUDA_CHECK;

  //weights[..., 1:-1] -- > 62
  dim3 dimBlock(256);
  dim3 dimGrid((batch_size + dimBlock.x - 1) / dimBlock.x);
  samplePDF_U<62,128><<<dimGrid, dimBlock>>>(width, batch_size, batch_offset,
    features, num_ray_samples, z_vals, weights, 64 , z_near, z_far, min_d_range, max_d_range);

  cudaDeviceSynchronize();
  CUDA_CHECK;

  dim3 dimBlock2((128 + 64));
  dim3 dimGrid2((batch_size * (128 + 64 ) + dimBlock2.x - 1) / dimBlock2.x);
  RayMarchFromCoarse<64,128, true><<<dimGrid2, dimBlock2>>>(width, batch_size, batch_offset,
    features, features_last, features_clean, z_step,
    noise_amplitude, num_ray_samples, depth_transform, z_near, z_far, min_d_range, max_d_range, z_vals,
    make_float3(vc_center_x, vc_center_y, vc_center_z), max_depth,
    encoding, num_dir_enc, num_pos_enc, dir_freq_bands, pos_freq_bands,
    weights, out_prev_size, true, make_float3(x,y,z), rotation_matrix);
  CUDA_CHECK;
}