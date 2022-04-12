


#include <iostream>

#include "StdStreamLog.h"


StdStreamLog::StdStreamLog()
  : log_errors(0),
    log_warnings(0)
{
}

void StdStreamLog::warning(const char* message, const char* file, int line)
{
  ++log_warnings;
  std::cout << file << '(' << line << "): warning: " << message << std::endl;
}

void StdStreamLog::warning(const std::string& message, const char* file, int line)
{
  ++log_warnings;
  std::cout << file << '(' << line << "): warning: " << message << std::endl;
}

void StdStreamLog::error(const char* message, const char* file, int line)
{
  ++log_errors;
  std::cout << file << '(' << line << "): error: " << message << std::endl;
}

void StdStreamLog::error(const std::string& message, const char* file, int line)
{
  ++log_errors;
  std::cout << file << '(' << line << "): error: " << message << std::endl;
}
