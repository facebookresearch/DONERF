


#ifndef INCLUDED_X11_GL_APPLICATION
#define INCLUDED_X11_GL_APPLICATION

#pragma once

#include "platform.h"

#include <GL/platform/InputHandler.h>
#include <GL/platform/Renderer.h>


namespace X11
{
	namespace GL
	{
		void run(::GL::platform::Renderer& renderer);
		void run(::GL::platform::Renderer& renderer, ::GL::platform::ConsoleHandler* console_handler);

		void quit();
	}
}

#endif  // INCLUDED_X11_GL_APPLICATION
