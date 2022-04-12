#include "../include/settings.h"

#include <algorithm>
#include <cmath>
#include <iostream>

Settings::Settings()
{
}
Settings::~Settings()
{
}


void Settings::init(argparse::ArgumentParser& program)
{
  model_path = program.get<std::string>("modelPath");

  auto size = program.get<std::vector<unsigned int>>("--size");
  width = size[0];
  height = size[1];
  total_size = width * height;

  if (program.is_used("--windowSize"))
  {
    auto window_size = program.get<std::vector<unsigned int>>("--windowSize");
    window_width = window_size[0];
    window_height = window_size[1];
  }
  else
  {
    window_width = width;
    window_height = height;
  }


  unsigned int n_batches = program.get<unsigned int>("--numberOfBatches");
  batch_size = std::ceil(total_size / (float) n_batches);

  if (program.is_used("--batchSize"))
  {
    int batch_size_arg = program.get<int>("--batchSize");
    batch_size = batch_size_arg <= 0 ? total_size : std::min((unsigned int) batch_size_arg, total_size); 
  }

  write_images = program.get<bool>("--writeImages");
  is_debug = program.get<bool>("--debug");
}
