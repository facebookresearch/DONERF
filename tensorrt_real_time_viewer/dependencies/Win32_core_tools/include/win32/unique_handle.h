


#ifndef INCLUDED_WIN32_UNIQUE_HANDLE
#define INCLUDED_WIN32_UNIQUE_HANDLE

#pragma once

#include <utility>

#include "platform.h"
#include "handle_deleter.h"


namespace Win32
{
	template <typename T, T null_value, class Deleter>
	class unique_handle : private Deleter
	{
	private:
		unique_handle(const unique_handle&);
		unique_handle& operator =(const unique_handle&);

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

		unique_handle()
			: h(null_value)
		{
		}

		explicit unique_handle(T handle)
			: h(handle)
		{
		}

		unique_handle(T handle, const Deleter& d)
			: Deleter(d),
			  h(handle)
		{
		}

		unique_handle(T handle, Deleter&& d)
			: Deleter(std::move(d)),
			  h(handle)
		{
		}

		unique_handle(unique_handle&& h)
			: Deleter(std::move(static_cast<Deleter&&>(h))),
			  h(h.h)
		{
			h.h = null_value;
		}

		~unique_handle()
		{
			close();
		}

		operator T() const { return h; }

		unique_handle& operator =(unique_handle&& h)
		{
			using std::swap;
			swap(*this, h);
			return *this;
		}

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

		friend void swap(unique_handle& a, unique_handle& b)
		{
			using std::swap;
			swap(a.h, b.h);
		}
	};

	template <typename T, T null_value, typename Deleter>
	const T unique_handle<T, null_value, Deleter>::null_value = null_value;
}

#endif  // INCLUDED_WIN32_UNIQUE_HANDLE
