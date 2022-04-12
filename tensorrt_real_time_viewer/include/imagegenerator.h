
#ifndef NRREALTIME_IMAGEGENERATOR_H
#define NRREALTIME_IMAGEGENERATOR_H

#pragma once

#include "NvInfer.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "../include/helper.h"
#include "../include/camera.h"
#include "../include/config.h"

class Settings;
class Config;

class ImageGenerator
{
private:
  Settings& settings;
  Config& config;

  nvinfer1::Dims mInputDims; 
  nvinfer1::Dims mOutputDims; 

  std::vector< std::shared_ptr<nvinfer1::ICudaEngine>> engines;
  std::vector< IGUniquePtr<nvinfer1::IExecutionContext>> contexts;

  std::vector<void*> d_outputs;
  std::vector<std::vector<void*>> inference_bindings;

  int in_idx_0, out_intermediate_idx_0, out_alpha_idx_0;
  int in_intermediate_idx_1, in_features_idx_1, in_alpha_idx_1, out_idx;

  void* d_intermediate;
  void* d_alpha;

  float s_inferenc1 = 0.0f;
  float s_inferenc2 = 0.0f;
  float s_fc1 = 0.0f;
  float s_fc2 = 0.0f;
  float s_rm = 0.0f;
  int s_num_total_samples = 0;

  int logging_interval = 100;
  int sample_count = 0;

  const int NUM_INTERMEDIATES = 3;

  void printNetwork(IGUniquePtr<nvinfer1::INetworkDefinition>& network);

  bool initEngine(const std::string& model_path, 
                  const std::string& feature_in, const std::string& feature_out, const std::string& encoding, 
                  int num_features, int num_out_features, int count, 
                  int batch_size, int num_raymarch_samples);

public:
  ImageGenerator(Settings& settings, Config& config);
  ~ImageGenerator();

  bool load(std::vector<int> num_in_features, std::vector<int> num_out_features);
  
  bool inference(Camera& camera, cudaSurfaceObject_t output_surf, int batch_size, int num_samples, 
                 std::vector<FeatureSet*>& feature_sets, std::vector<Encoding>& encodings);
                 
  bool writeImage(const samplesCommon::BufferManager& buffers);
};

#endif