
#ifndef NRREALTIME_INTEROPRENDERBUFFER_H
#define NRREALTIME_INTEROPRENDERBUFFER_H

#pragma once

#include "buffermanager.h"

#include <GL/gl.h>
#include <cuda_gl_interop.h>

#include <vector>

class InteropRenderbuffer : public BufferManager
{
private:
  GLuint fb[2];
  GLuint rb[2];

  cudaGraphicsResource* d_fb_res[2];
  cudaArray* d_fb_ar[2];
  cudaSurfaceObject_t d_fb_surf[2];

  unsigned int idx;

  std::vector <std::vector<void*>> inference_mappings;

public:
  InteropRenderbuffer();
  ~InteropRenderbuffer();

  bool init(unsigned int width, unsigned int height) override;
  void resize(unsigned int width, unsigned int height) override;
  unsigned int swap() override;
  void blit(unsigned int target_width, unsigned int target_height) override;
  void clear() override;
  void map() override;
  void unmap() override;
};

#endif