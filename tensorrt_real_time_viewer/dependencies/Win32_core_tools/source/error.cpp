


#include "memory.h"

#include "unicode.h"

#include "error.h"
#include "com_error.h"


namespace
{
	std::wstring formatMessage(DWORD error)
	{
		WCHAR* buffer;
		FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM, nullptr, error, 0, reinterpret_cast<LPWSTR>(&buffer), 0, nullptr);
		Win32::unique_heap_ptr<WCHAR> msg(buffer);
		return msg.get();
	}
}

namespace Win32
{
	std::string error::message() const
	{
		return narrow(formatMessage(error_code));
	}
}

namespace COM
{
	std::string error::message() const
	{
		return narrow(formatMessage(hresult));
	}
}
