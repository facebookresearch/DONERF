


#ifndef INCLUDED_WIN32_UNICODE
#define INCLUDED_WIN32_UNICODE

#pragma once

#include <string>

#include "platform.h"


namespace Win32
{
	std::basic_string<WCHAR> widen(const char* string, size_t length);
	std::basic_string<WCHAR> widen(const char* string);
	std::basic_string<WCHAR> widen(const std::string& string);
	
	std::string narrow(const WCHAR* string, size_t length);
	std::string narrow(const WCHAR* string);
	std::string narrow(const std::basic_string<WCHAR>& string);
}

using Win32::widen;
using Win32::narrow;

#endif	// INCLUDED_WIN32_UNICODE
