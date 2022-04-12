


#ifndef INCLUDED_LOG_H
#define INCLUDED_LOG_H

#pragma once

#include <string>

#include "interface.h"


class INTERFACE Log
{
protected:
  Log() {}
  Log(const Log&) {}
  ~Log() {}
  Log& operator =(const Log&) { return *this; }
public:
  virtual int errors() const = 0;
  virtual int warnings() const = 0;

  virtual void warning(const char* message, const char* file, int line) = 0;
  virtual void warning(const std::string& message, const char* file, int line) = 0;
  virtual void error(const char* message, const char* file, int line) = 0;
  virtual void error(const std::string& message, const char* file, int line) = 0;
};


#endif  // INCLUDED_LOG_H
