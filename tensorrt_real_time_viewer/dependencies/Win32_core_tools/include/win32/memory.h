


#ifndef INCLUDED_WIN32_MEMORY
#define INCLUDED_WIN32_MEMORY

#pragma once

#include <memory>

#include "unique_handle.h"
#include "scoped_handle.h"


namespace Win32
{
	struct DefaultHeapDeleter
	{
		void operator ()(void* p) const
		{
			HeapFree(GetProcessHeap(), 0U, p);
		}
	};

	template <typename T, typename Del = DefaultHeapDeleter>
	using unique_heap_ptr = std::unique_ptr<T, Del>;


	struct GlobalFreeDeleter
	{
		void operator ()(HGLOBAL hmem) const
		{
			GlobalFree(hmem);
		}
	};

	using scoped_hglobal = scoped_handle<HGLOBAL, 0, GlobalFreeDeleter>;
	using unique_hglobal = unique_handle<HGLOBAL, 0, GlobalFreeDeleter>;


	struct LocalFreeDeleter
	{
		void operator ()(HLOCAL hmem) const
		{
			LocalFree(hmem);
		}
	};

	using scoped_hlocal = scoped_handle<HLOCAL, 0, LocalFreeDeleter>;
	using unique_hlocal = unique_handle<HLOCAL, 0, LocalFreeDeleter>;
}

#endif  // INCLUDED_WIN32_MODULE_HANDLE
