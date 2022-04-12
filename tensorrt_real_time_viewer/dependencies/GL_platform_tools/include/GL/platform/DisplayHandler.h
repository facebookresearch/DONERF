


#ifndef INCLUDED_GL_PLATFORM_DISPLAY_HANDLER
#define INCLUDED_GL_PLATFORM_DISPLAY_HANDLER

#pragma once

#include "interface.h"


namespace GL
{
	namespace platform
	{
		class INTERFACE DisplayHandler
		{
		protected:
			DisplayHandler() {}
			DisplayHandler(const DisplayHandler&) {}
			DisplayHandler& operator =(const DisplayHandler&) { return *this; }
			~DisplayHandler() {}
		public:
			virtual void close() = 0;
			virtual void destroy() = 0;

			virtual void move(int x, int y) = 0;
			virtual void resize(int width, int height) = 0;
		};
	}
}

#endif  // INCLUDED_GL_PLATFORM_DISPLAY_HANDLER
