


#include <cassert>
#include <stdexcept>

#include <win32/window_handle.h>

#include "Win32GLContext.h"

#include <GL/gl.h>
#include "wglext.h"


namespace
{
	Win32::GL::unique_glcoreContext initglcoreContext(HDC hdc, HGLRC hglrc)
	{
		wglMakeCurrent(hdc, hglrc);
		Win32::GL::unique_glcoreContext ctx(glcoreContextInit());
		wglMakeCurrent(0, 0);
		return ctx;
	}
}

namespace Win32
{
	namespace GL
	{
		void setPixelFormat(HDC hdc, int depth_buffer_bits, int stencil_buffer_bits, bool stereo)
		{
			PIXELFORMATDESCRIPTOR pfd = {
				sizeof(PIXELFORMATDESCRIPTOR),
				1U,
				PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER | (stereo ? PFD_STEREO : 0U),
				PFD_TYPE_RGBA,
				24U,
				8U, 0U, 8U, 0U, 8U, 0U, 8U, 0U,
				0U,
				0U, 0U, 0U, 0U,
				static_cast<BYTE>(depth_buffer_bits),
				static_cast<BYTE>(stencil_buffer_bits),
				0U,
				PFD_MAIN_PLANE,
				0U,
				0U, 0U, 0U
			};

			int pf = ChoosePixelFormat(hdc, &pfd);

			if (pf == 0)
				throw std::runtime_error("ChoosePixelFormat() failed");

			if (SetPixelFormat(hdc, pf, nullptr) == FALSE)
				throw std::runtime_error("SetPixelFormat() failed");
		}

		unique_hglrc createContext(HDC hdc, int version_major, int version_minor, bool debug)
		{
			unique_hglrc dummy_context(wglCreateContext(hdc));

			if (dummy_context == 0)
				throw std::runtime_error("dummy context creation failed");

			wglMakeCurrent(hdc, dummy_context);

			auto wglCreateContextAttribsARB = reinterpret_cast<PFNWGLCREATECONTEXTATTRIBSARBPROC>(wglGetProcAddress("wglCreateContextAttribsARB"));

			if (wglCreateContextAttribsARB == nullptr)
				throw std::runtime_error("wglCreateContextAttribsARB() not supported");


			static const int attribs[] = {
				WGL_CONTEXT_MAJOR_VERSION_ARB, version_major,
				WGL_CONTEXT_MINOR_VERSION_ARB, version_minor,
				WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
				WGL_CONTEXT_FLAGS_ARB, debug ? WGL_CONTEXT_DEBUG_BIT_ARB : 0,
				0
			};

			unique_hglrc context(wglCreateContextAttribsARB(hdc, 0, attribs));

			if (context == 0)
				throw std::runtime_error("wglCreateContextAttribsARB() failed");

			return context;
		}

		Context::Context(HDC hdc, int version_major, int version_minor, bool debug)
			: hglrc(createContext(hdc, version_major, version_minor, debug)),
			  ctx(initglcoreContext(hdc, hglrc))
		{
		}
	}
}
