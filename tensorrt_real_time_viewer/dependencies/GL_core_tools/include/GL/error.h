


#ifndef INCLUDED_GL_ERROR
#define INCLUDED_GL_ERROR

#pragma once

#include <exception>

#include <GL/gl.h>


namespace GL
{
	class error : public std::exception
	{
	private:
		GLenum error_code;
	public:
		error(GLenum error_code);

#if defined(_MSC_VER) && _MSC_VER <= 1800
		const char* what() const;
#else
		const char* what() const noexcept;
#endif
	};

	inline void checkError()
	{
		if (GLenum err = glGetError())
			throw GL::error(err);
	}
}

#endif  // INCLUDED_GL_ERROR
