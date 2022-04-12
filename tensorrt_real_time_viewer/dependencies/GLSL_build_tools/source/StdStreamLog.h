


#ifndef INCLUDED_STDSTREAMLOG_H
#define INCLUDED_STDSTREAMLOG_H

#pragma once

#include "Log.h"


class StdStreamLog : public virtual Log
{
protected:
  StdStreamLog(const StdStreamLog&);
  StdStreamLog& operator =(const StdStreamLog&);

  int log_warnings;
  int log_errors;

public:
  StdStreamLog();

  int errors() const { return log_errors; }
  int warnings() const { return log_warnings; }

  void warning(const char* message, const char* file, int line);
  void warning(const std::string& message, const char* file, int line);
  void error(const char* message, const char* file, int line);
  void error(const std::string& message, const char* file, int line);
};


#endif  // INCLUDED_STDSTREAMLOG_H
