


#ifndef INCLUDED_WIN32_GL_CONTEXT
#define INCLUDED_WIN32_GL_CONTEXT

#pragma once

#include <utility>
#include <memory>

#include <win32/platform.h>
#include <win32/unique_handle.h>
#include <win32/glcore.h>


namespace Win32
{
	namespace GL
	{
		struct wglDeleteContextDeleter
		{
			void operator ()(HGLRC context)
			{
				wglDeleteContext(context);
			}
		};

		typedef unique_handle<HGLRC, 0, wglDeleteContextDeleter> unique_hglrc;

		void setPixelFormat(HDC hdc, int depth_buffer_bits, int stencil_buffer_bits, bool stereo = false);
		unique_hglrc createContext(HDC hdc, int version_major, int version_minor, bool debug = false);


		struct glcoreContextDestroyDeleter
		{
			void operator ()(const glcoreContext* ctx)
			{
				glcoreContextDestroy(ctx);
			}
		};

		typedef std::unique_ptr<const glcoreContext, glcoreContextDestroyDeleter> unique_glcoreContext;


		class Context
		{
			template <class SurfaceType>
			friend class context_scope;
		private:
			unique_hglrc hglrc;
			unique_glcoreContext ctx;

		public:
			Context(const Context&) = delete;
			Context& operator =(const Context&) = delete;

			Context(HDC hdc, int version_major, int version_minor, bool debug = false);

			Context(Context&& c)
				: hglrc(std::move(c.hglrc)),
				  ctx(std::move(c.ctx))
			{
			}

			Context& operator =(Context&& c)
			{
				using std::swap;
				swap(hglrc, c.hglrc);
				swap(ctx, c.ctx);
				return *this;
			}
		};


		template <class SurfaceType>
		struct SurfaceTypeTraits;

		template <class SurfaceType>
		class context_scope : private SurfaceTypeTraits<SurfaceType>::ContextScopeState
		{
		private:
			HDC hdc;
			HGLRC hglrc;
			const glcoreContext* ctx;

			HDC hdc_restore;
			HGLRC hglrc_restore;
			const glcoreContext* ctx_restore;

			void makeCurrent()
			{
				Win32::checkError(wglMakeCurrent(hdc, hglrc) != TRUE);
				glcoreContextMakeCurrent(ctx);
			}

		public:
			context_scope(const context_scope&) = delete;
			context_scope& operator =(const context_scope&) = delete;

			context_scope(Context& context, SurfaceType& surface)
				: SurfaceTypeTraits<SurfaceType>::ContextScopeState(surface),
				  hdc(openHDC()),
				  hglrc(context.hglrc),
				  ctx(context.ctx.get()),
				  hdc_restore(wglGetCurrentDC()),
				  hglrc_restore(wglGetCurrentContext()),
				  ctx_restore(glcoreContextGetCurrent())
			{
				makeCurrent();
			}

			~context_scope()
			{
				Win32::checkError(wglMakeCurrent(hdc_restore, hglrc_restore) != TRUE);
				glcoreContextMakeCurrent(ctx_restore);
				closeHDC(hdc);
			}

			void bind(Context& context, SurfaceType& surface)
			{
				typename SurfaceTypeTraits<SurfaceType>::ContextScopeState old_state(surface);

				HDC old_hdc = hdc;

				using std::swap;
				swap(*this, old_state);
				hdc = openDC();

				makeCurrent();

				old_state.releaseDC(old_hdc);
			}

			void setSwapInterval(int interval)
			{
				wglSwapIntervalEXT(interval);
			}

			void swapBuffers()
			{
				SwapBuffers(hdc);
			}
		};
	}
}

#endif  // INCLUDED_WIN32_GL_CONTEXT
