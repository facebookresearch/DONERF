


#include <cassert>
#include <stdexcept>

#include "X11GLContext.h"


namespace X11
{
	namespace GL
	{
		Context createContext(::Display* display, GLXFBConfig fb_config, int version_major, int version_minor, bool debug)
		{
			static struct glx_ext_loader
			{
				PFNGLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribs;

				glx_ext_loader()
					: glXCreateContextAttribs(reinterpret_cast<decltype(glXCreateContextAttribs)>(glXGetProcAddress(reinterpret_cast<const GLubyte*>("glXCreateContextAttribsARB"))))
				{
					if (glXCreateContextAttribs == nullptr)
						throw std::runtime_error("glXCreateContextAttribsARB() not supported.");
				}
			} glx_ext;

			static const int attribs[] = {
				GLX_CONTEXT_MAJOR_VERSION_ARB, version_major,
				GLX_CONTEXT_MINOR_VERSION_ARB, version_minor,
				GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
				GLX_CONTEXT_FLAGS_ARB, debug ? GLX_CONTEXT_DEBUG_BIT_ARB : 0,
				None
			};

			GLXContext context = glx_ext.glXCreateContextAttribs(display, fb_config, 0, True, attribs);

			if (context == 0)
				throw std::runtime_error("glXCreateContextAttribs() failed");

			return Context(display, context);
		}
	}
}
