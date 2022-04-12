


#include "error.h"


namespace
{
	const char* genErrorMessage(GLenum error)
	{
		switch (error)
		{
		case GL_INVALID_ENUM:
			return "GL_INVALID_ENUM";
		case GL_INVALID_VALUE:
			return "GL_INVALID_VALUE";
		case GL_INVALID_OPERATION:
			return "GL_INVALID_OPERATION";
		case GL_INVALID_FRAMEBUFFER_OPERATION:
			return "GL_INVALID_FRAMEBUFFER_OPERATION";

		}

		return "unknown error code";
	}
}

namespace GL
{
	error::error(GLenum error_code)
		: error_code(error_code)
	{
	}

#if defined(_MSC_VER) && _MSC_VER <= 1800
	const char* error::what() const
#else
	const char* error::what() const noexcept
#endif
	{
		return genErrorMessage(error_code);
	}
}
