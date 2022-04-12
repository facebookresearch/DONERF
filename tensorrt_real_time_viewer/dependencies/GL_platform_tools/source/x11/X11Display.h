


#ifndef INCLUDED_PLATFORM_X11_DISPLAY
#define INCLUDED_PLATFORM_X11_DISPLAY

#pragma once

#include <utility>

#include "platform.h"


namespace X11
{
	class Display
	{
	private:
		::Display* display;

	public:
		Display(const Display&) = delete;
		Display& operator =(const Display&) = delete;

		Display(::Display* display)
			: display(display)
		{
		}

		Display(Display&& d)
			: display(d.display)
		{
			d.display = 0;
		}

		~Display();

		Display& operator =(Display&& d)
		{
			using std::swap;
			swap(display, d.display);
			return *this;
		}

		operator ::Display*() const { return display; }
	};

	Display openDisplay();
}

#endif  // INCLUDED_PLATFORM_X11_DISPLAY
