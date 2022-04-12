


#ifndef INCLUDED_WIN32_GL_APPLICATION
#define INCLUDED_WIN32_GL_APPLICATION

#pragma once

#include <win32/platform.h>

#include <GL/platform/InputHandler.h>
#include <GL/platform/Renderer.h>


namespace Win32
{
	namespace GL
	{
		void run(::GL::platform::Renderer& renderer);
		void run(::GL::platform::Renderer& renderer, ::GL::platform::ConsoleHandler* console_handler);

		void quit();
	}
}

#endif  // INCLUDED_WIN32_GL_APPLICATION
