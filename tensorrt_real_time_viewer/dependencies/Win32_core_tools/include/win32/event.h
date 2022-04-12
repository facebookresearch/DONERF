


#ifndef INCLUDED_WIN32_EVENT_HANDLE
#define INCLUDED_WIN32_EVENT_HANDLE

#pragma once

#include "unique_handle.h"
#include "scoped_handle.h"


namespace Win32
{
	typedef scoped_handle<HANDLE, 0, CloseHandleDeleter> scoped_hevent;
	typedef unique_handle<HANDLE, 0, CloseHandleDeleter> unique_hevent;

	unique_hevent createEvent();
}

#endif  // INCLUDED_WIN32_EVENT_HANDLE
