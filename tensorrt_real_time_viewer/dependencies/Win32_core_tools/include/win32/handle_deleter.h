


#ifndef INCLUDED_WIN32_HANDLE_DELETER
#define INCLUDED_WIN32_HANDLE_DELETER

#pragma once

#include "platform.h"


namespace Win32
{
	struct CloseHandleDeleter
	{
		void operator ()(HANDLE h) const
		{
			CloseHandle(h);
		}
	};
}

#endif  // INCLUDED_WIN32_HANDLE_DELETER
