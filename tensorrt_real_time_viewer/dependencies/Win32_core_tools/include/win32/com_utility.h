


#ifndef INCLUDED_WIN32_COM_UTILITY
#define INCLUDED_WIN32_COM_UTILITY

#pragma once

#include "platform.h"
#include "com_error.h"
#include "com_ptr.h"


namespace COM
{
	template <typename T>
	inline com_ptr<T> QueryInterface(IUnknown* obj)
	{
		void* p;
		checkError(obj->GetParent(getIID<T>(), &p));
		return make_com_ptr(static_cast<T*>(p));
	}
}

#endif  // INCLUDED_WIN32_COM_UTILITY
