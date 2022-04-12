
#ifndef NRREALTIME_HELPER_H
#define NRREALTIME_HELPER_H

#pragma once

#include <vector>
#include <memory>

#ifndef PATH_SEPARATOR
  #if defined(WIN32) || defined(_WIN32) 
  #define PATH_SEPARATOR "\\" 
  #else 
  #define PATH_SEPARATOR "/" 
  #endif 
#endif

struct InferDeleter
{
  template <typename T>
  void operator()(T* obj) const
  {
    if (obj)
    {
      obj->destroy();
    }
  }
};

template <typename T>
using IGUniquePtr = std::unique_ptr<T, InferDeleter>;

#endif 