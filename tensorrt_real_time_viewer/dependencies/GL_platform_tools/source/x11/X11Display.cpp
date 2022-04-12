


#include "X11Display.h"


namespace X11
{
	Display openDisplay()
	{
		return XOpenDisplay(nullptr);
	}

	Display::~Display()
	{
		XCloseDisplay(display);
	}
}
