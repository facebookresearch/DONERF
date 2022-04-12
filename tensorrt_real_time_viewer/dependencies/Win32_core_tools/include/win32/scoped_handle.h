


#ifndef INCLUDED_WIN32_SCOPED_HANDLE
#define INCLUDED_WIN32_SCOPED_HANDLE

#pragma once

#include <utility>

#include "platform.h"
#include "handle_deleter.h"


namespace Win32
{
	template <typename T, T null_value, class Deleter>
	class scoped_handle : private Deleter
	{
	private:
		scoped_handle(const scoped_handle&);
		scoped_handle& operator =(const scoped_handle&);

		T h;

		void close()
		{
			if (h != null_value)
				Deleter::operator ()(h);
		}

	public:
		typedef T handle_type;
		typedef Deleter deleter_type;

		static const T null_value;

		explicit scoped_handle(T handle)
			: h(handle)
		{
		}

		scoped_handle(T handle, const Deleter& d)
			: Deleter(d),
			  h(handle)
		{
		}

		scoped_handle(T handle, Deleter&& d)
			: Deleter(std::move(d)),
			  h(handle)
		{
		}

		~scoped_handle()
		{
			close();
		}

		operator T() const { return h; }

		void reset(T handle = null_value)
		{
			close();
			h = handle;
		}

		T release()
		{
			T temp = h;
			h = null_value;
			return temp;
		}

		friend void swap(scoped_handle& a, scoped_handle& b)
		{
			using std::swap;
			swap(a.h, b.h);
		}
	};

	template <typename T, T null_value, typename Deleter>
	const T scoped_handle<T, null_value, Deleter>::null_value = null_value;
}

#endif  // INCLUDED_WIN32_SCOPED_HANDLE
