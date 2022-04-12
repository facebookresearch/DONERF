
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


template<int N_DEPTHS>
__global__ void setSpherePosDirBatchedUnrolledNoEnc(float* __restrict features, 
  const float* __restrict features_clean, const float* __restrict rot_mat, 
  float px, float py, float pz, float cx, float cy, float cz, 
  float radius, float min_d, float max_d, int batch_offset, int batch_size)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int ray_sample_idx = idx & (N_DEPTHS - 1);
  unsigned int ray_idx = idx / N_DEPTHS;

  if ((ray_idx >= batch_size) || (ray_sample_idx >= N_DEPTHS))
    return;

  const float3 fc = reinterpret_cast<const float3*>(features_clean)[(batch_offset + ray_idx) * 2];
  const float3 rt1 = reinterpret_cast<const float3*>(rot_mat)[0];
  const float3 rt2 = reinterpret_cast<const float3*>(rot_mat)[1];
  const float3 rt3 = reinterpret_cast<const float3*>(rot_mat)[2];

  float3 d;
  d.x = fc.x * rt1.x + fc.y * rt1.y + fc.z * rt1.z;
  d.y = fc.x * rt2.x + fc.y * rt2.y + fc.z * rt2.z;
  d.z = fc.x * rt3.x + fc.y * rt3.y + fc.z * rt3.z;

  d = normalize(d);

  float3 np = raySphereIntersect(make_float3(px,py,pz), d, make_float3(cx,cy,cz), radius);

  float3* r_feat = reinterpret_cast<float3*>(features);
  int idx_r = (ray_idx) * (2 + N_DEPTHS);

  float step = 1.0f / N_DEPTHS;
  float cur_step = step / 2.0f + step * ray_sample_idx;

  r_feat[idx_r + 2 + ray_sample_idx] = np + d * logtToWorld(cur_step, min_d, max_d);

  if (ray_sample_idx != 0)
    return;
  
  r_feat[idx_r] = d;
  r_feat[idx_r + 1] = np;
}


__global__ void samplePDF(int batch_size, int num_ray_samples, float* z_vals, float* depth, int num_depths)
{
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= batch_size)
    return;

  int idx = id;// x + y * size;

  float depth_sum = 0.0f;

  float current_depth = 0.0f;

  int cdf_size = num_depths + 1;
  float cdfs_[129];
  float depths_[129];

  int n_samples = num_ray_samples + 2;

  for (int k = 0; k < num_depths; k++)
  {
    current_depth = sigmoid(depth[idx * num_depths + k]) + 1e-5f; // depth transform;   weight += 1e-5
    depths_[k] = current_depth;
    depth_sum += current_depth; //  torch.sum(weights, -1, keepdim=True)
  }

  float cdf = 0.0f;
  float u = 0.0f;
  float u_step = 1.0f / (n_samples - 1);

  int below[10];
  int above[10];
  int ind_c = 0;

  for (int k = 0; k < cdf_size; k++)
  {
    float pdf = 0;
    if (k < num_depths)
      pdf = depths_[k] / depth_sum;

    while (cdf > u && ind_c < 10)
    {
      below[ind_c] = max(0, k - 1);
      above[ind_c] = min(cdf_size - 1, k);

      ind_c++;
      u += u_step;
    }
    cdfs_[k] = cdf;
    cdf += pdf;
  }

  for (int l = ind_c; l < 10; l++)
  {
    below[l] = max(0, cdf_size - 2);
    above[l] = min(cdf_size - 1, cdf_size - 1);
  }
  u = 0;
  for (int l = 0; l < n_samples; l++)
  {
    float cdf_g0 = cdfs_[below[l]];
    float cdf_g1 = cdfs_[above[l]];

    float bin_g0 = getBin(num_depths, below[l]); // num_bins = num_depths+1 ; torch.linspace(0., 1., 128+1, device="cpu", dtype=torch.float32)
    float bin_g1 = getBin(num_depths, above[l]);

    float denom = cdf_g1 - cdf_g0;
    if (denom < 1e-5f)
      denom = 1.0f;

    float t = (u - cdf_g0) / denom;
    float sample = bin_g0 + t * (bin_g1 - bin_g0);

    if (l > 0 && l < num_ray_samples + 1)
      z_vals[idx * num_ray_samples + l - 1] = sample;
    u += u_step;
  }
}

__global__ void rayMarchFromPoses(int size, int batch_size, int batch_offset, 
  float* features, float* features_last, float z_step, float noise_amplitude, int num_ray_samples, 
  int depth_transform, float z_near, float z_far, float min_d_range, float max_d_range, float* z_vals,
  float3 center, float max_depth, int num_dir_freqs, int num_pos_freqs, 
  float* dir_freq_bands, float* pos_freq_bands, float* depth, int num_depths)
{
  unsigned int feature_idx = threadIdx.x;
  unsigned int ray_idx = blockIdx.x / num_ray_samples;
  unsigned int sample_idx = blockIdx.x % num_ray_samples;

  if (ray_idx >= batch_size || feature_idx >= num_pos_freqs + 1 || sample_idx >= num_ray_samples)
    return;

  float3* features_r = reinterpret_cast<float3*>(features);
  float3* features_last_r = reinterpret_cast<float3*>(features_last);

  float3 p = features_last_r[ray_idx * (128 + 2) + 1];
  float3 d = features_last_r[ray_idx * (128 + 2) + 0];

  d = normalize(d);

  int n_features_r = (1 + num_dir_freqs * 2 + 1 + num_pos_freqs * 2);
  int feature_idx_r = ray_idx * num_ray_samples * n_features_r;

  float z_val = logtToWorld(z_vals[ray_idx * num_ray_samples + sample_idx], min_d_range, max_d_range);

  float3 p_ = p + d * z_val;

  float3 np = normalizationInverseSqrtDistCentered(p_, center, max_depth);

  int p_start_r = feature_idx_r + sample_idx * n_features_r;
  int d_start_r = feature_idx_r + sample_idx * n_features_r + 1 + num_pos_freqs * 2;


  if (feature_idx == 0)
  {
    features_r[p_start_r] = np;
    features_r[d_start_r] = d;
    z_vals[ray_idx * num_ray_samples + sample_idx] = z_val;
    return;
  }
  feature_idx -= 1;

  float fbv = pos_freq_bands[feature_idx];
  float3 v_f = np * fbv;
  features_r[p_start_r + 1 + feature_idx * 2 + 0] = make_float3(sin(v_f.x), sin(v_f.y), sin(v_f.z));
  features_r[p_start_r + 1 + feature_idx * 2 + 1] = make_float3(cos(v_f.x), cos(v_f.y), cos(v_f.z));

  if (feature_idx >= num_dir_freqs)
    return;

  v_f = d * fbv;
  features_r[d_start_r + 1 + feature_idx * 2 + 0] = make_float3(sin(v_f.x), sin(v_f.y), sin(v_f.z));
  features_r[d_start_r + 1 + feature_idx * 2 + 1] = make_float3(cos(v_f.x), cos(v_f.y), cos(v_f.z));
}

__global__ void nerf_raw_2_output(float* shading_output, cudaSurfaceObject_t out_fb_surf, 
  float* z_vals, float* features, int size, int num_samples, int batch_size, int batch_offset)
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

  float nr = 0.0f;
  float ng = 0.0f;
  float nb = 0.0f;
  
  for (int k = 0; k < num_samples; k++) 
  {
    // dists = z_vals[..., 1:] - z_vals[..., :-1];    
    // dists = torch.cat([dists, (torch.ones(1, device = raw.device) * 1e10).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    float dist = (k < num_samples - 1)
      ? z_vals[start_2 + k + 1] - z_vals[start_2 + k]
      : 1e10f;

    // rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    float r = sigmoid(shading_output[start + 0 + k * 4]);
    float g = sigmoid(shading_output[start + 1 + k * 4]);
    float b = sigmoid(shading_output[start + 2 + k * 4]);

    float alpha = raw2alpha(shading_output[start + 3 + k * 4], dist);

    // weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device = raw.device), 1. - alpha + 1e-10], -1), -1)[:,:-1]
    float prod =  last_prod * (1.0f - alpha + 1e-10f);
    float weight = alpha * last_prod;  

    // rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    nr = nr + weight * r;
    ng = ng + weight * g;
    nb = nb + weight * b;

    last_prod = prod;
  }

  uchar4 data;
  data.x = clamp(nr, 0.0f, 1.0f) * 255.0f;
  data.y = clamp(ng, 0.0f, 1.0f) * 255.0f;
  data.z = clamp(nb, 0.0f, 1.0f) * 255.0f;
  data.w = 255.0f;
  surf2Dwrite(data, out_fb_surf, x_ * 4, y_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HOST CODE 

void updateSpherePosDirBatchedUnrolledNoEnc(float* features, float* features_clean, float x, float y, float z, float* rotation_matrix, 
  int width, int height, int num_dir_enc, int num_pos_enc, float* dir_freq_bands, float* pos_freq_bands,
  float center_x, float center_y, float center_z, float radius, int additional_samples, float min_d_range, float max_d_range,
  int batch_offset, int batch_size)
{
  const int N_DEPTHS = 128;
  const int N_RAYS_PER_BLOCK = 4;
  dim3 dimBlock(N_DEPTHS * N_RAYS_PER_BLOCK);
  dim3 dimGrid((batch_size + N_RAYS_PER_BLOCK - 1) / N_RAYS_PER_BLOCK);

  assert(additional_samples == N_DEPTHS && "use constant for better performance");

  setSpherePosDirBatchedUnrolledNoEnc<N_DEPTHS><<<dimGrid, dimBlock>>>(features, features_clean, rotation_matrix, 
    x, y, z, center_x, center_y, center_z , radius, min_d_range, max_d_range, batch_offset, batch_size);

  CUDA_CHECK;
}

void updateRayMarchFromPosesSplit(float* features, float* features_clean, float* features_last, 
  float x, float y, float z, float* rotation_matrix, int width, int height,
  int num_dir_enc, int num_pos_enc, float* dir_freq_bands, float* pos_freq_bands,
  float z_step, float noise_amplitude, int num_ray_samples, int depth_transform, 
  float z_near, float z_far, float min_d_range, float max_d_range, float* z_vals,
  float vc_center_x, float vc_center_y, float vc_center_z, float max_depth,
  int batch_offset, int batch_size,  float* depth, float* cdf, int num_depths)
{
  dim3 dimBlock(1024);
  dim3 dimGrid((batch_size + dimBlock.x - 1) / dimBlock.x);
  samplePDF<<<dimGrid, dimBlock>>>(batch_size, num_ray_samples, z_vals, depth, num_depths);
  
  cudaDeviceSynchronize();
  CUDA_CHECK;
  assert(depth_transform && "ERROR: need to change function because of depth transform");

  int n_l_features = num_pos_enc + 1;
  dim3 dimBlock2(n_l_features );
  dim3 dimGrid2(batch_size * num_ray_samples);

  rayMarchFromPoses<<<dimGrid2, dimBlock2>>>(width, batch_size, batch_offset, features, features_last, 
    z_step, noise_amplitude, num_ray_samples, depth_transform, 
    z_near, z_far, min_d_range, max_d_range, z_vals,
    make_float3(vc_center_x, vc_center_y, vc_center_z), max_depth,
    num_dir_enc, num_pos_enc, dir_freq_bands, pos_freq_bands, depth, num_depths);

  cudaDeviceSynchronize();
}


void copyResultRaymarch(void* shading_output, cudaSurfaceObject_t fb_surf, int width, int height, 
  float* z_vals, float* features, int num_samples, int batch_size, int batch_offset)
{
  dim3 dimBlock(16);
  dim3 dimGrid((batch_size + dimBlock.x - 1) / dimBlock.x);
  nerf_raw_2_output<<<dimGrid, dimBlock>>>((float*) shading_output, fb_surf, z_vals, features, width, num_samples, batch_size, batch_offset);
  
  CUDA_CHECK;
}