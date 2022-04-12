


#ifndef INCLUDED_GL_PLATFORM_RENDERER
#define INCLUDED_GL_PLATFORM_RENDERER

#pragma once

#include "interface.h"


namespace GL
{
	namespace platform
	{
		class INTERFACE Renderer
		{
		protected:
			Renderer() {}
			Renderer(const Renderer&) {}
			Renderer& operator =(const Renderer&) { return *this; }
			~Renderer() {}
		public:
			virtual void render() = 0;
		};
	}
}

#endif  // INCLUDED_GL_PLATFORM_RENDERER
