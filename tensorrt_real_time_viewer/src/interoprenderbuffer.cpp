#include "../include/interoprenderbuffer.h"

#include <cstring>


InteropRenderbuffer::InteropRenderbuffer() 
  : fb{0, 0}, rb{0, 0}, d_fb_res{nullptr, nullptr}, d_fb_ar{nullptr, nullptr}, d_fb_surf{0, 0}, idx{0}
{
}

InteropRenderbuffer::~InteropRenderbuffer() 
{
  for (int i = 0; i < 2; i++) 
  {
    if (d_fb_res[i] != nullptr)
      cudaGraphicsUnregisterResource(d_fb_res[i]);
  }

  if(rb[0] != 0)
    glDeleteRenderbuffers(2, rb);
  if (fb[0] != 0)
    glDeleteFramebuffers(2, fb);

}

bool InteropRenderbuffer::init(unsigned int w, unsigned int h)
{
  width = w;
  height = h;
  idx = 0;
  
  glCreateRenderbuffers(2, &rb[0]);
  glCreateFramebuffers(2, &fb[0]);

  glNamedFramebufferRenderbuffer(fb[0], GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,  rb[0]);
  glNamedFramebufferRenderbuffer(fb[1], GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,  rb[1]);
  this->resize(w, h);

  d_curr_buffer = d_fb_ar[idx];
  d_curr_buffer_surf = d_fb_surf[idx];

  return true;
}

void InteropRenderbuffer::resize(unsigned int w, unsigned int h)
{
  width = w;
  height = h;

  if (!inference_mappings.empty())
    inference_mappings.clear();

  for (int i = 0; i < 2; i++) 
  {
    if (d_fb_res[i] != NULL)
      cudaGraphicsUnregisterResource(d_fb_res[i]);

    glNamedRenderbufferStorage(rb[i], GL_RGBA8, width, height);
    
    cudaGraphicsGLRegisterImage(&d_fb_res[i], rb[i], GL_RENDERBUFFER,
      cudaGraphicsRegisterFlagsSurfaceLoadStore |
      cudaGraphicsRegisterFlagsWriteDiscard);
  }

  cudaGraphicsMapResources(2, &d_fb_res[0], 0);
  for (int index = 0; index < 2; index++)
  {
    cudaGraphicsSubResourceGetMappedArray(&d_fb_ar[index], d_fb_res[index], 0, 0);
    std::vector<void*> map;
    map.push_back((void*)&d_fb_ar[index]);
    inference_mappings.push_back(map);

    if (d_fb_surf[index])
      cudaDestroySurfaceObject(d_fb_surf[index]);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_fb_ar[index];
    cudaCreateSurfaceObject(&d_fb_surf[index], &resDesc);
  }
  cudaGraphicsUnmapResources(2, &d_fb_res[0], 0);
}

void InteropRenderbuffer::blit(unsigned int target_width, unsigned int target_height)
{
  glBlitNamedFramebuffer(fb[idx], 0, 0, 0, width, height, 0, target_height, target_width, 0, GL_COLOR_BUFFER_BIT, target_width > width ? GL_LINEAR : GL_NEAREST);
}

void InteropRenderbuffer::clear()
{
  GLfloat clear_color[] = { 1.0f, 0.0f, 1.0f, 1.0f };
  glClearNamedFramebufferfv(fb[idx], GL_COLOR, 0, clear_color);
}

unsigned int InteropRenderbuffer::swap()
{
  idx = (idx + 1) % 2;
  
  d_curr_buffer = d_fb_ar[idx];
  d_curr_buffer_surf = d_fb_surf[idx];

  return idx;
}

void InteropRenderbuffer::map()
{
  cudaGraphicsMapResources(1, &d_fb_res[idx]);
}
void InteropRenderbuffer::unmap()
{
  cudaGraphicsUnmapResources(1, &d_fb_res[idx]);
}