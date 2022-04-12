


#ifndef INCLUDED_GL_BUFFER
#define INCLUDED_GL_BUFFER

#pragma once

#include <GL/gl.h>

#include "unique_name.h"


namespace GL
{
	struct BufferObjectNamespace
	{
		static GLuint gen();
		static void del(GLuint name);
	};

	typedef unique_name<BufferObjectNamespace> Buffer;
}

#endif  // INCLUDED_GL_BUFFER
