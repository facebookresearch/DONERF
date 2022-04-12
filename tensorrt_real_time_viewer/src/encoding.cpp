#include "../include/encoding.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

Encoding::Encoding()
{
}

Encoding::~Encoding()
{
}

void genPosEncFreqs(std::vector<float>& bands, int num_freqs)
{
  for (int x = 0; x < num_freqs; x++)
  {
    float val = 0;
    val = (float)x;
    bands.push_back((float)pow(2, val));
  }
}


bool Encoding::load(std::string type_, int num_pos_freqs_, int num_dir_freqs_, int add_samples)
{
  type = type_;

  num_pos_freqs = 0;
  num_dir_freqs = 0;

  additional_samples = add_samples;


  if (type == "nerf")
  {
    num_pos_freqs = num_pos_freqs_;
    num_dir_freqs = num_dir_freqs_;
    genPosEncFreqs(pos_freq_bands, num_pos_freqs);
    genPosEncFreqs(dir_freq_bands, num_dir_freqs);

    cudaMalloc((void**)&(d_pos_freq_bands), num_pos_freqs * sizeof(float));
    cudaMalloc((void**)&(d_dir_freq_bands), num_dir_freqs * sizeof(float));

    cudaMemcpy(d_pos_freq_bands, &pos_freq_bands[0], num_pos_freqs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dir_freq_bands, &dir_freq_bands[0], num_dir_freqs * sizeof(float), cudaMemcpyHostToDevice);
  }
  if (type == "fourier")
  {
  }

  return true;
}



