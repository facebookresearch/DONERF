


#ifndef INCLUDED_WIN32_ERROR
#define INCLUDED_WIN32_ERROR

#pragma once

#include <string>

#include "platform.h"


namespace Win32
{
	class error
	{
	private:
		DWORD error_code;
	public:
		explicit error(DWORD error_code)
			: error_code(error_code)
		{
		}

		std::string message() const;
	};

	inline void checkError(bool condition)
	{
		if (condition)
			throw error(GetLastError());
	}
}

#endif  // INCLUDED_WIN32_ERROR
