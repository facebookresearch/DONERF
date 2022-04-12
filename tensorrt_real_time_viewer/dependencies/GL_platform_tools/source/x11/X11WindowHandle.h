


#ifndef INCLUDED_PLATFORM_X11_WINDOW_HANDLE
#define INCLUDED_PLATFORM_X11_WINDOW_HANDLE

#pragma once

#include <utility>

#include "platform.h"


namespace X11
{
	class WindowHandle
	{
	private:
		::Display* disp;
		::Window window;

	public:
		WindowHandle()
			: disp(nullptr),
			  window(0)
		{
		}

		WindowHandle(::Display* display, ::Window window)
			: disp(display),
			  window(window)
		{
		}

		WindowHandle(const WindowHandle&) = delete;
		WindowHandle& operator =(const WindowHandle&) = delete;

		WindowHandle(WindowHandle&& w)
			: disp(w.disp),
			  window(w.window)
		{
			w.window = 0;
		}

		~WindowHandle()
		{
			if (window)
				XDestroyWindow(disp, window);
		}

		WindowHandle& operator =(WindowHandle&& w)
		{
			using std::swap;
			disp = w.disp;
			swap(window, w.window);
			return *this;
		}

		operator ::Window() const { return window; }
		
		::Display* display() const { return disp; }
	};

	inline WindowHandle createWindow(::Display* display, ::Window parent, int x, int y, unsigned int width, unsigned int height, unsigned int border_width, int depth, unsigned int window_class, ::Visual* visual, unsigned long valuemask, XSetWindowAttributes* attributes)
	{
		return WindowHandle(display, XCreateWindow(display, parent, x, y, width, height, border_width, depth, window_class, visual, valuemask, attributes));
	}
}

#endif  // INCLUDED_PLATFORM_X11_WINDOW_HANDLE
