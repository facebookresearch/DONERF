


#ifndef INCLUDED_GL_PLATFORM_APPLICATION
#define INCLUDED_GL_PLATFORM_APPLICATION

#pragma once

#include "Renderer.h"


#if defined(_WIN32)
#include "../../../source/win32/Win32GLApplication.h"
namespace GL
{
	namespace platform
	{
		using Win32::GL::run;
		using Win32::GL::quit;
	}
}
#elif defined(__gnu_linux__)
#include "../../../source/x11/X11GLApplication.h"
namespace GL
{
	namespace platform
	{
		using X11::GL::run;
		using X11::GL::quit;
	}
}
#else
#error "platform not supported."
#endif

#endif  // INCLUDED_GL_PLATFORM_APPLICATION
