


#ifndef INCLUDED_WIN32_MODULE_HANDLE
#define INCLUDED_WIN32_MODULE_HANDLE

#pragma once

#include "unique_handle.h"
#include "scoped_handle.h"


namespace Win32
{
	struct FreeLibraryDeleter
	{
		void operator ()(HMODULE module) const
		{
			FreeLibrary(module);
		}
	};

	typedef scoped_handle<HMODULE, 0, FreeLibraryDeleter> scoped_hmodule;
	typedef unique_handle<HMODULE, 0, FreeLibraryDeleter> unique_hmodule;
}

#endif  // INCLUDED_WIN32_MODULE_HANDLE
