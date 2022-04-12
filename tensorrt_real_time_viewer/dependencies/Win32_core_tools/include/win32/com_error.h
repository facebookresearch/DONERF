


#ifndef INCLUDED_WIN32_COM_ERROR
#define INCLUDED_WIN32_COM_ERROR

#pragma once

#include <string>

#include "platform.h"


namespace COM
{
	class error
	{
	private:
		HRESULT hresult;
	public:
		explicit error(HRESULT hresult)
			: hresult(hresult)
		{
		}
		
		std::string message() const;
	};

	inline void checkError(HRESULT hresult)
	{
		if (FAILED(hresult))
			throw error(hresult);
	}
}

using COM::checkError;

#endif  // INCLUDED_WIN32_COM_ERROR
