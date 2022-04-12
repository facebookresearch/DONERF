


#ifndef INCLUDED_WIN32_FILE
#define INCLUDED_WIN32_FILE

#pragma once

#include <utility>
#include <string>

#include "platform.h"
#include "error.h"
#include "unique_handle.h"
#include "scoped_handle.h"


namespace Win32
{
	typedef scoped_handle<HANDLE, INVALID_HANDLE_VALUE, CloseHandleDeleter> scoped_hfile;
	typedef unique_handle<HANDLE, INVALID_HANDLE_VALUE, CloseHandleDeleter> unique_hfile;

	inline unique_hfile createFile(const wchar_t* file_name, DWORD access, DWORD share_mode, DWORD create, DWORD attributes)
	{
		HANDLE h = CreateFileW(file_name, access, share_mode, nullptr, create, attributes, 0);
		checkError(h == INVALID_HANDLE_VALUE);
		return unique_hfile(h);
	}

	inline unique_hfile createFile(const char* file_name, DWORD access, DWORD share_mode, DWORD create, DWORD attributes)
	{
		HANDLE h = CreateFileA(file_name, access, share_mode, nullptr, create, attributes, 0);
		checkError(h == INVALID_HANDLE_VALUE);
		return unique_hfile(h);
	} 

	inline void read(HANDLE file, char* buffer, size_t size)
	{
		DWORD bytes_read;
		checkError(ReadFile(file, buffer, static_cast<DWORD>(size), &bytes_read, nullptr) != TRUE || bytes_read != size);
	}

	inline void write(HANDLE file, const char* buffer, size_t size)
	{
		DWORD bytes_written;
		checkError(WriteFile(file, buffer, static_cast<DWORD>(size), &bytes_written, nullptr) != TRUE || bytes_written != size);
	}
}

#endif  // INCLUDED_WIN32_FILE
