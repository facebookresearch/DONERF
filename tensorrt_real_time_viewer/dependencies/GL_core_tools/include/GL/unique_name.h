


#ifndef INCLUDED_GL_UNIQUE_NAME
#define INCLUDED_GL_UNIQUE_NAME

#pragma once

#include <utility>

#include <GL/gl.h>


namespace GL
{
	template <typename Namespace>
	class unique_name
	{
	private:
		GLuint name;

	public:
		unique_name(const unique_name&) = delete;
		unique_name& operator =(const unique_name&) = delete;

		unique_name()
			: name(Namespace::gen())
		{
		}

		explicit unique_name(GLuint name)
			: name(name)
		{
		}

		unique_name(unique_name&& n)
			: name(n.name)
		{
			n.name = 0U;
		}

		~unique_name()
		{
			if (name != 0U)
				Namespace::del(name);
		}

		operator GLuint() const { return name; }

		unique_name& operator =(unique_name&& n)
		{
			using std::swap;
			swap(this->name, n.name);
			return *this;
		}

		void reset(GLuint new_name = 0U)
		{
			if (name != 0U)
				Namespace::del(name);
			name = new_name;
		}

		GLuint release()
		{
			GLuint name = this->name;
			this->name = 0U;
			return name;
		}

		friend void swap(unique_name& a, unique_name& b)
		{
			using std::swap;
			swap(a.name, b.name);
		}
	};
}

#endif  // INCLUDED_GL_UNIQUE_NAME
