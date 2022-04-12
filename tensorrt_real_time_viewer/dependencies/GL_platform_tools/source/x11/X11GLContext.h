


#ifndef INCLUDED_PLATFORM_X11_GL_CONTEXT
#define INCLUDED_PLATFORM_X11_GL_CONTEXT

#pragma once

#include <utility>
#include <stdexcept>

#include "platform.h"
#define GLX_GLXEXT_PROTOTYPES
#include "glxext.h"


namespace X11
{
	namespace GL
	{
		class Context
		{
		private:
			::Display* display;
			GLXContext context;

		public:
			Context(const Context&) = delete;
			Context& operator =(const Context&) = delete;

			Context()
				: display(nullptr),
				  context(0)
			{
			}

			Context(::Display* display, GLXContext context)
				: display(display),
				  context(context)
			{
			}

			Context(Context&& c)
				: display(c.display),
				  context(c.context)
			{
				c.context = 0;
			}

			~Context()
			{
				if (context)
					glXDestroyContext(display, context);
			}

			Context& operator =(Context&& c)
			{
				using std::swap;
				swap(display, c.display);
				swap(context, c.context);
				return *this;
			}

			operator GLXContext() const { return context; }
		};

		Context createContext(::Display* display, GLXFBConfig fb_config, int version_major, int version_minor, bool debug = false);


		template <class SurfaceType>
		struct SurfaceTypeTraits;

		template <class SurfaceType>
		class context_scope : private SurfaceTypeTraits<SurfaceType>::ContextScopeState
		{
		private:
			::Display* display;
			GLXDrawable drawable;
			GLXContext context;

			::Display* display_restore;
			GLXDrawable drawable_restore;
			GLXContext context_restore;

			void makeCurrent()
			{
				if (!glXMakeCurrent(display, drawable, context))
					throw std::runtime_error("glXMakeCurrent() failed");
			}

		public:
			context_scope(const context_scope&) = delete;
			context_scope& operator =(const context_scope&) = delete;

			context_scope(Context& context, SurfaceType& surface)
				: SurfaceTypeTraits<SurfaceType>::ContextScopeState(surface),
				  display(SurfaceTypeTraits<SurfaceType>::ContextScopeState::display(surface)),
				  drawable(SurfaceTypeTraits<SurfaceType>::ContextScopeState::drawable(surface)),
				  context(context),
				  display_restore(glXGetCurrentDisplay()),
				  drawable_restore(glXGetCurrentDrawable()),
				  context_restore(glXGetCurrentContext())
			{
				makeCurrent();
			}

			~context_scope()
			{
				glXMakeCurrent(display_restore, drawable_restore, context_restore);
			}

			void bind(SurfaceType& surface)
			{
				typename SurfaceTypeTraits<SurfaceType>::ContextScopeState old_state(surface);
				using std::swap;
				swap(*this, old_state);
				
				display = SurfaceTypeTraits<SurfaceType>::ContextScopeState::display(surface);
				drawable = SurfaceTypeTraits<SurfaceType>::ContextScopeState::drawable(surface);

				makeCurrent();
			}

			void setSwapInterval(int interval)
			{
				glXSwapIntervalEXT(display, drawable, interval);
			}

			void swapBuffers()
			{
				glXSwapBuffers(display, drawable);
			}
		};
	}
}

#endif  // INCLUDED_PLATFORM_X11_GL_CONTEXT
