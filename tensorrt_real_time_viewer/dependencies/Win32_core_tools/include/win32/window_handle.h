


#ifndef INCLUDED_WIN32_WINDOW_HANDLE
#define INCLUDED_WIN32_WINDOW_HANDLE

#pragma once

#include "unique_handle.h"
#include "scoped_handle.h"


namespace Win32
{
	struct DestroyWindowDeleter
	{
		void operator ()(HWND hwnd) const
		{
			DestroyWindow(hwnd);
		}
	};

	typedef scoped_handle<HWND, 0, DestroyWindowDeleter> scoped_hwnd;
	typedef unique_handle<HWND, 0, DestroyWindowDeleter> unique_hwnd;
}

#endif  // INCLUDED_WIN32_WINDOW_HANDLE
