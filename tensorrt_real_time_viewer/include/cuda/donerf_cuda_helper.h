
#ifndef DONERF_HELPER_H
#define DONERF_HELPER_H

#include "cuda_runtime.h"
#include "helper_math.h"


inline __device__ float sigmoid(float a)
{
  return 1.0f / (1.0f + expf(-a));
}
inline __device__  float relu(float raw)
{
  return max(0.0f, raw);
}
inline __device__  float raw2alpha(float raw, float dist)
{
  return 1.0f - expf(-relu(raw) * dist);
}
inline __device__  double raw2alpha(double raw, double dist)
{
  return 1.0f - expf(-relu(raw) * dist);
}

inline __device__ float logtFromWorld(float depth, float min_d, float max_d)
{
  float max_v = max_d - min_d;
  return logf(depth - min_d + 1.0f) / logf(max_v + 1);
}
inline __device__ float logtToWorld(float depth, float min_d, float max_d)
{
  float max_v = max_d - min_d;
  return powf(max_v + 1, depth) - 1.0f + min_d;
}

inline __device__ float lintFromWorld(float depth, float min_d, float max_d)
{
  return (depth - min_d) / (max_d - min_d);
}
inline __device__ float lintToWorld(float depth, float min_d, float max_d)
{
  return depth * (max_d - min_d) + min_d;
}


inline __device__ float getBin(int disc_steps, int bin_id)
{
  float step = 1.0f / (disc_steps);
  return bin_id * step;
}

inline __device__ float getBinMid(int disc_steps, int bin_id, float z_near, float z_far , float min_d_range, float max_d_range)
{
  float step =  ( (z_far - z_near) / disc_steps);
  float b1 = logtToWorld(z_near + step * ((float) bin_id ), min_d_range, max_d_range);
  float b2 = logtToWorld(z_near + step * ((float) min( bin_id + 1, disc_steps ) ), min_d_range, max_d_range);
  return  (b1 + b2) *0.5f ;
}

inline __device__ float getBinMidSimple(int disc_steps, int bin_id)
{
  float step = 1.0f / (disc_steps);
  return (bin_id + .5f) * step;
}


inline __device__ float3 normalizationInverseSqrtDistCentered(float3 x_in_world, float3  view_cell_center, float max_depth)
{
  float3 localized = x_in_world - view_cell_center; 
  float l = sqrtf(length(localized));
  return localized / (sqrtf(max_depth) * l);
}

inline __device__ float3 raySphereIntersect(float3 o, float3 u, float3 c, float r)
{
  float3 o_m_c = o - c;
  float d = dot(u, o_m_c);

  float omc_l = length(o_m_c);
  float delta = d * d - (omc_l * omc_l - r * r);
  float sqrt_delta = sqrt(max(delta, 0.0f));

  float dis = -d + sqrt_delta;

  return o + (u * dis);
}

#endif