
#ifndef NRREALTIME_SETTINGS_H
#define NRREALTIME_SETTINGS_H

#pragma once

#include <string>
#include <vector>

#include "argparse/argparse.hpp"

class Settings
{
private:

public:
  Settings();
  ~Settings();

  void init(argparse::ArgumentParser& program);

  // Config properties
  std::string model_path;

  unsigned int width;
  unsigned int height;
  unsigned int total_size;
  unsigned int window_width;
  unsigned int window_height;
  unsigned int batch_size;

  bool write_images;
  bool is_debug;
};

#endif