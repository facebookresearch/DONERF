


#ifndef INCLUDED_GL_PLATFORM_CONTEXT
#define INCLUDED_GL_PLATFORM_CONTEXT

#pragma once


#if defined(_WIN32)
#include "../../../source/win32/Win32GLContext.h"
namespace GL
{
	namespace platform
	{
		using Win32::GL::Context;
		using Win32::GL::context_scope;
	}
}
#elif defined(__gnu_linux__)
#include "../../../source/x11/X11GLContext.h"
namespace GL
{
	namespace platform
	{
		using X11::GL::Context;
		using X11::GL::context_scope;
	}
}
#else
#error "platform not supported."
#endif

#endif  // INCLUDED_GL_PLATFORM_CONTEXT
