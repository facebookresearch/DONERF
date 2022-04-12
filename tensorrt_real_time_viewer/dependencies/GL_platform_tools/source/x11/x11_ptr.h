


#ifndef INCLUDED_PLATFORM_X11_PTR
#define INCLUDED_PLATFORM_X11_PTR

#pragma once

#include "platform.h"


namespace X11
{
	struct deleter
	{
		void operator ()(void* ptr) const
		{
			XFree(ptr);
		}
	};
}

#endif  // INCLUDED_PLATFORM_X11_PTR
