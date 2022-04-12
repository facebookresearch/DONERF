


#ifndef INCLUDED_GL_SCOPED_NAME
#define INCLUDED_GL_SCOPED_NAME

#pragma once

#include <utility>

#include <GL/gl.h>


namespace GL
{
	template <typename Namespace>
	class scoped_name : private Namespace
	{
	private:
		scoped_name(const scoped_name&);
		scoped_name& operator =(const scoped_name&);

		GLuint id;

	public:
		explicit scoped_name(GLuint id = 0U)
			: id(id)
		{
		}

		~scoped_name()
		{
			if (h != 0U)
				close(h);
		}

		operator GLuint() const { return id; }

		unique_name& operator =(unique_name&& h)
		{
			swap(*this, h);
			return *this;
		}

		void reset(T unique_handle = 0U)
		{
			if (h != 0U)
				close(h);
			h = unique_handle;
		}

		T release()
		{
			T unique_handle = h;
			h = 0U;
			return unique_handle;
		}

		friend void swap(unique_name& a, unique_name& b)
		{
			using std::swap;
			swap(a.h, b.h);
		}
	};
}

#endif  // INCLUDED_GL_SCOPED_NAME
