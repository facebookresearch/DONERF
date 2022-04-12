
#ifndef NRREALTIME_FEATURESET_H
#define NRREALTIME_FEATURESET_H

#pragma once

#include <string>
#include <cuda_runtime.h>
#include <glm/vec3.hpp>

class Encoding;
class Config;
class Settings;

class FeatureSet
{
public:
  FeatureSet(Settings& settings, Config& config);
  ~FeatureSet() {}

  float* getFeatures() { return d_features; }

  std::string type;
  float vc_center[3];
  float vc_size[3];
  float vc_radius;

protected:
  Settings& settings;
  Config& config;

  float* d_features = nullptr;

  Encoding* enc;
};

class SpherePosDir : public FeatureSet
{
private:
  int additional_samples = 0;
  float max_drange = 0;
  float min_drange = 0;

public:
  SpherePosDir(Settings& settings, Config& config) :FeatureSet(settings, config) {}
  ~SpherePosDir() {}

  bool create(int width, int height, int batch_size, Encoding* enc_);
  void updateFeaturesBatched(glm::vec3 point, float* rot_mat, float* features_clean, 
                             int width, int height, int batch_offset, int batch_size);
};

class RayMarchFromPoses : public FeatureSet
{
private:
  float z_step = 0;
  float noise_amplitude = 0;
  int num_ray_samples = 0;
  int depth_transform = 0;
  float min_d = 0;
  float max_d = 0;

  float max_drange = 0;
  float min_drange = 0;

  float max_depth = 0;

  float vc_x = 0;
  float vc_y = 0;
  float vc_z = 0;

  float* d_z_vals = 0;
  float* d_cdf = 0;

  // Offset of the samples of each ray. 
  // Rays not consecutively in z_vals, but samples of each ray are (+ in order of depth)!
  int* d_ray_idx_per_z_val = nullptr;
  int* d_sample_idx_per_z_val = nullptr;
  int* d_ray_offsets = nullptr;
  int* d_ray_n_samples = nullptr; // number of samples of each ray
  
  std::string sampler = "";

public:
  RayMarchFromPoses(Settings& settings, Config& config) : FeatureSet(settings, config) {}
  ~RayMarchFromPoses() {}

  int getNumRaySamples() { return num_ray_samples; }
  float* getZVals() { return d_z_vals; }

  int* getRayOffsets() { return d_ray_offsets; }
  int* getRayNumSamples() { return d_ray_n_samples; }


  bool create(int width, int height, int batch_size, Encoding* enc_, int count);
  int updateFeatures(glm::vec3 point, float* rot_mat, float* features_clean, float* features_last, 
                      int width, int height, int batch_offset, int batch_size, float* depth, int num_depths);

  void raymarch(void* shading_output, cudaSurfaceObject_t fb_surf, int batch_size, int batch_offset, int width, int height, float* features);
};

class RayMarchFromCoarse: public FeatureSet
{
private:
  float z_step = 0;
  float noise_amplitude = 0;
  int num_ray_samples = 0;
  int num_ray_samples_pre = 0;
  int depth_transform = 0;
  float min_d = 0;
  float max_d = 0;

  float max_drange = 0;
  float min_drange = 0;

  float max_depth = 0;

  float vc_x = 0;
  float vc_y = 0;
  float vc_z = 0;

  float* d_z_vals = nullptr;
  float* d_cdf = nullptr;
  float* d_weights = nullptr;

  std::string sampler = "";

public:
  RayMarchFromCoarse(Settings& settings, Config& config) : FeatureSet(settings, config) {}
  ~RayMarchFromCoarse() {}

  int getNumRaySamples() { return num_ray_samples; }
  float* getZVals() { return d_z_vals; }

  bool create(int width, int height, int batch_size, Encoding* enc_, int count);
  void updateFeatures(glm::vec3 point, float* rot_mat, float* features_clean, float* features_last, 
                      int width, int height, int batch_offset, int batch_size, float* prev_z_vals,
                      float* prev_weights, int prev_num_weights);
};

#endif