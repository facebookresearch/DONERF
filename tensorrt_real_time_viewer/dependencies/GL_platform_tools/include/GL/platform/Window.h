


#ifndef INCLUDED_GL_PLATFORM_WINDOW
#define INCLUDED_GL_PLATFORM_WINDOW

#pragma once

#include "DisplayHandler.h"
#include "InputHandler.h"


#if defined(_WIN32)
#include "../../../source/win32/Win32GLWindow.h"
namespace GL
{
	namespace platform
	{
		using Win32::GL::Window;
	}
}
#elif defined(__gnu_linux__)
#include "../../../source/x11/X11GLWindow.h"
namespace GL
{
	namespace platform
	{
		using X11::GL::Window;
	}
}
#else
#error "platform not supported."
#endif

#endif  // INCLUDED_GL_PLATFORM_WINDOW
