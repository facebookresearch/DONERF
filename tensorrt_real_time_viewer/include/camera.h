
#ifndef NRREALTIME_CAMERA_H
#define NRREALTIME_CAMERA_H

#pragma once

#include <string>
#include <GL/gl.h>
#include <glm/vec2.hpp> 
#include <glm/vec3.hpp> 
#include <glm/mat4x4.hpp> 

#include "helper.h"
#include "featureset.h"
#include "settings.h"

class Encoding;
class Featureset;
class Config;

class Camera
{
private:

  float* features = nullptr;
  float* features_clean = nullptr;

  float* d_rotation_matrix = nullptr;
  float* d_features_clean = nullptr;

  glm::vec3 pos = { 0, 0, 0 };
  glm::vec3 dir = { 0, 0, 0 };
  glm::vec3 right = { 0, 0, 0 };
  glm::vec3 up = { 0, 0, 0 };

  float yaw = 0;
  float pitch = 0;

  float fov = 0;
  float focal = 0;

  glm::vec3 m_min = { 0, 0, 0 };
  glm::vec3 m_max = { 0, 0, 0 };

  unsigned int num_features = 0;

  bool pos_changed = false;
  bool rot_changed = false;

  void UpdateFeaturePos();
  void UpdateFeatureRot();
  
  Encoding* enc = nullptr;
  int feature_encoding = 0;

  FeatureSet* input_features = nullptr;

  Settings& settings;
  Config& config;

  //glm::vec3 move_dir{ 0, 0, 0 };
  int move_fwd = 0;
  int move_right = 0;
  int move_up = 0;

public:
   Camera(Settings& settings, Config& config);
  ~Camera();

  bool init(std::string feature_type, float fov, Encoding* enc_);
  float* getHostFeatureVector() { return features; }
  float* getDeviceFeatureVector() { return input_features->getFeatures(); }
  int getFeatureVectorSize() { return settings.total_size * sizeof(float) * (3 + 3); }
  Encoding* getEncoding() { return enc; }
  FeatureSet* getFeatureSet() { return input_features; }


  float* getRotMatrix() { return d_rotation_matrix; }
  float* getFeaturesClean() { return d_features_clean; }
  glm::vec3 getPosition() { return pos; }

  void MovementKeyPressed(unsigned char type);
  void MovementKeyReleased(unsigned char type);

  void MouseDrag(const glm::ivec2& offset);

  void UpdateFeaturesBatch(int batch_offset, int batch_size);
};

#endif