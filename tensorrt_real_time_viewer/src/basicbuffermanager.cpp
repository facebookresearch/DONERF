#include "../include/basicbuffermanager.h"

#include "../include/cuda/donerf_cuda_kernels.cuh"

#include <cstring>


BasicBufferManager::~BasicBufferManager()
{
  if (initialized)
  {
    cudaDestroySurfaceObject(d_curr_buffer_surf);
    CUDA_CHECK;
  
    cudaFreeArray(d_curr_buffer);
    CUDA_CHECK;
  }    
}

bool BasicBufferManager::init(unsigned int w, unsigned int h)
{
  width = w;
  height = h;
  
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
  cudaMallocArray(&d_curr_buffer, &channelDesc, width, height, cudaArraySurfaceLoadStore);
  CUDA_CHECK;

  cudaResourceDesc surfRes;
  memset(&surfRes, 0, sizeof(cudaResourceDesc));
  surfRes.resType = cudaResourceTypeArray;
  surfRes.res.array.array = d_curr_buffer;

  cudaCreateSurfaceObject(&d_curr_buffer_surf, &surfRes);
  CUDA_CHECK;

  this->resize(w, h);
  initialized = true;

  return true;
}

void BasicBufferManager::resize(unsigned int w, unsigned int h)
{
  width = w;
  height = h;
}

void BasicBufferManager::clear()
{
}