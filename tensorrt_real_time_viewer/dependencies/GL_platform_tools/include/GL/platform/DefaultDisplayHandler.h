


#ifndef INCLUDED_GL_PLATFORM_DEFAULT_DISPLAY_HANDLER
#define INCLUDED_GL_PLATFORM_DEFAULT_DISPLAY_HANDLER

#pragma once

#include "DisplayHandler.h"


namespace GL
{
	namespace platform
	{
		class DefaultDisplayHandler : public virtual DisplayHandler
		{
		protected:
			DefaultDisplayHandler() {}
			DefaultDisplayHandler(const DefaultDisplayHandler&) : DisplayHandler() {}
			DefaultDisplayHandler& operator =(const DefaultDisplayHandler&) { return *this; }
			~DefaultDisplayHandler() {}
		public:
			void close();
			void destroy();

			void move(int x, int y);
			void resize(int width, int height);
		};
	}
}

#endif  // INCLUDED_GL_PLATFORM_DEFAULT_DISPLAY_HANDLER
