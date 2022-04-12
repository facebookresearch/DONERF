


#ifndef INCLUDED_WIN32_GL_WINDOW
#define INCLUDED_WIN32_GL_WINDOW

#pragma once

#include <win32/platform.h>
#include <win32/error.h>
#include <win32/window_handle.h>

#include <GL/platform/DisplayHandler.h>
#include <GL/platform/InputHandler.h>
#include <GL/platform/Renderer.h>

#include "Win32GLContext.h"


namespace Win32
{
	namespace GL
	{
		void setPixelFormat(HWND hwnd, int depth_buffer_bits, int stencil_buffer_bits, bool stereo = false);
		Context createContext(HWND hwnd, int version_major, int version_minor, bool debug = false);


		class Window
		{
			friend class WindowContextScopeState;
		private:
			unique_hwnd hwnd;

			WINDOWPLACEMENT windowed_placement;

			::GL::platform::DisplayHandler* display_handler;
			::GL::platform::MouseInputHandler* mouse_input;
			::GL::platform::KeyboardInputHandler* keyboard_input;

			LRESULT WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

			static unique_hwnd createWindow(Window& wnd, DWORD dwExStyle, LPCWSTR lpWindowName, DWORD dwStyle, int X, int Y, int nWidth, int nHeight, HWND hWndParent, HMENU hMenu);
			static unique_hwnd createWindow(Window& wnd, const char* title, int x, int y, int width, int height);
			static unique_hwnd createWindow(Window& wnd, const char* title);

		public:
			Window(const Window&) = delete;
			Window& operator =(const Window&) = delete;

			Window(const char* title, int width, int height, int depth_buffer_bits = 0, int stencil_buffer_bits = 0, bool stereo = false, bool fullscreen = false);
			Window(const char* title, int x, int y, int width, int height, int depth_buffer_bits = 0, int stencil_buffer_bits = 0, bool stereo = false, bool fullscreen = false);
			Window(const char* title, const WINDOWPLACEMENT& placement, int depth_buffer_bits = 0, int stencil_buffer_bits = 0, bool stereo = false);

			void savePlacement(WINDOWPLACEMENT& placement) const;
			void place(const WINDOWPLACEMENT& placement);

			void title(const char* title);

			void resize(int width, int height);

			void toggleFullscreen();

			Context createContext(int version_major, int version_minor, bool debug = false);

			void attach(::GL::platform::DisplayHandler* display_handler);
			void attach(::GL::platform::MouseInputHandler* mouse_input);
			void attach(::GL::platform::KeyboardInputHandler* keyboard_input);
		};


		class WindowContextScopeState
		{
		private:
			HWND hwnd;

		protected:
			HDC openHDC()
			{
				return GetDC(hwnd);
			}

			void closeHDC(HDC hdc)
			{
				ReleaseDC(hwnd, hdc);
			}

		public:
			WindowContextScopeState(HWND hwnd)
				: hwnd(hwnd)
			{
			}

			WindowContextScopeState(Window& window)
				: hwnd(window.hwnd)
			{
			}
		};

		template <>
		struct SurfaceTypeTraits<Window>
		{
			typedef WindowContextScopeState ContextScopeState;
		};

		template <>
		struct SurfaceTypeTraits<HWND>
		{
			typedef WindowContextScopeState ContextScopeState;
		};
	}
}

#endif  // INCLUDED_WIN32_GL_WINDOW
