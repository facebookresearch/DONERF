
#ifndef NRREALTIME_BASICBUFFERMANAGER_H
#define NRREALTIME_BASICBUFFERMANAGER_H

#pragma once

#include "buffermanager.h"

/**
 * @brief Simple BufferManager with a single array and surface as buffer. 
 *  Used for profiling and debugging (because of NSight profiling issues with cuda_gl_interop)
 */
class BasicBufferManager : public BufferManager
{
private:
  bool initialized = false;

public:
  BasicBufferManager() {};
  ~BasicBufferManager();

  bool init(unsigned int width, unsigned int height) override;
  void resize(unsigned int width, unsigned int height) override;

  void clear() override;
};

#endif