
#ifndef NRREALTIME_ENCODING_H
#define NRREALTIME_ENCODING_H

#pragma once

#include <string>
#include <vector>

class Encoding
{
private:
  std::vector<float> pos_freq_bands;
  std::vector<float> dir_freq_bands;
  std::vector<float> b;

  int num_pos_freqs;
  int num_dir_freqs;

  int additional_samples = 0;

public:
  std::string type;
  float* d_pos_freq_bands;
  float* d_dir_freq_bands;
  

  Encoding();
  ~Encoding();

  bool load(std::string type_, int num_pos_freqs_, int num_dir_freqs_, int add_samples=0);

  int getNumPosFreqs() { return (type != "none") * num_pos_freqs; }
  int getNumDirFreqs() { return (type != "none") * num_dir_freqs; }
  int getNumFeatures() { return 3 + 3 * getNumPosFreqs() * 2 + 3 + 3 * getNumDirFreqs() * 2 + additional_samples * ( 3 + 3 * getNumPosFreqs() * 2); }


};

#endif
