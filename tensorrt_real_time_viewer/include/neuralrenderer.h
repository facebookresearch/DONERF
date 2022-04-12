
#ifndef NRREALTIME_NEURALRENDERER_H
#define NRREALTIME_NEURALRENDERER_H

#pragma once

#include <chrono>
#include <string>
#include <glm/vec2.hpp> 

#include "buffermanager.h"
#include "encoding.h"
#include "helper.h"
#include "imagegenerator.h"

#include "GL/platform/Window.h"
#include "GL/platform/Context.h"
#include "GL/platform/DefaultDisplayHandler.h"


class Camera;
class Config;
class Settings;
class FeatureSet;

class NeuralRenderer : public GL::platform::Renderer, public GL::platform::DisplayHandler
{
private:
  Settings& settings;
  Config& config;
  GL::platform::Window& window;
  Camera& camera;
  ImageGenerator img_gen;

  GL::platform::Context context;
  GL::platform::context_scope<GL::platform::Window> context_scope;

  BufferManager* buffer_manager;

  glm::ivec2 window_size;
  std::vector<Encoding> encodings;
  std::vector<FeatureSet*> feature_sets;

  void swapBuffers();
  void writeImageToFile();

public:
  NeuralRenderer(Settings& settings, Config& config, GL::platform::Window& window, Camera& camera,
                 int opengl_version_major, int opengl_version_minor);

  ~NeuralRenderer();

  bool init();

  virtual void render();

  virtual void move(int x, int y) override;
  virtual void resize(int width, int height) override;
  virtual void close() override;
  virtual void destroy() override;

  std::vector<FeatureSet*>& getFeatureSet() { return feature_sets; }
  std::vector<Encoding>& getEncoding() { return encodings; }

  std::chrono::steady_clock::time_point last;
};

#endif