
#ifndef NRREALTIME_BUFFERMANAGER_H
#define NRREALTIME_BUFFERMANAGER_H

#pragma once

#include <GL/gl.h>

#include <cuda_runtime.h>

class BufferManager
{
protected:
  cudaArray* d_curr_buffer = nullptr;
  cudaSurfaceObject_t d_curr_buffer_surf = 0;

  unsigned int width = 0;
  unsigned int height = 0;

public:
  BufferManager() {};
  virtual ~BufferManager() {};

  virtual bool init(unsigned int width, unsigned int height) = 0;
  virtual void resize(unsigned int width, unsigned int height) = 0;

  virtual unsigned int swap() { return 0; };
  virtual void blit(unsigned int, unsigned int) {};
  virtual void clear() {};
  virtual void map() {};
  virtual void unmap() {};

  cudaArray* getCurrentBuffer() { return d_curr_buffer; }
  cudaSurfaceObject_t getCurrentBufferSurface() { return d_curr_buffer_surf; }
};

#endif