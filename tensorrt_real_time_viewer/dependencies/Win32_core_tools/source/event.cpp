


#include "error.h"
#include "event.h"


namespace Win32
{
	unique_hevent createEvent()
	{
		HANDLE h = CreateEventW(nullptr, FALSE, FALSE, nullptr);
		if (h == 0)
			throw error(GetLastError());
		return unique_hevent(h);
	}
}
