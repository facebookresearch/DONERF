
#ifndef NRREALTIME_CONFIG_H
#define NRREALTIME_CONFIG_H

#pragma once

#include <string>
#include <vector>

class Config
{
private:
  std::string file_dir;
  std::string file_name;

  std::vector<int> tc1;
  std::vector<int> tc2;

  void store(std::string key, std::string value);

public:
  Config();
  ~Config();

  bool load(std::string file_name);
  bool loadDatasetInfo(std::string base_data_dir_);


  // ----------------------------------------------
  // Model Description

  std::vector<std::vector<float>> posEncArgs;
  std::vector<std::string> posEnc;
  std::vector<std::string> inFeatures;
  std::vector<std::string> outFeatures;

  std::vector<int> numRaymarchSamples;
  std::vector<std::string> rayMarchSampler;
  std::vector<std::string> rayMarchNormalization;
  std::vector<std::string> activation;
  std::vector<float> rayMarchSamplingStep;
  std::vector<float> rayMarchSamplingNoise;
  std::vector<int> raySampleInput;
  std::string depthTransform;
  std::vector<float> zNear;
  std::vector<float> zFar;

  // ----------------------------------------------
  // Camera Config

  float fov;
  float max_depth;
  std::vector<float> viewcellCenter;
  std::vector<float> viewcellSize;
  std::vector<float> depthRange;
  
  // ----------------------------------------------
};

#endif