


#ifndef INCLUDED_GL_VERTEX_ARRAY
#define INCLUDED_GL_VERTEX_ARRAY

#pragma once

#include <GL/gl.h>

#include "unique_name.h"


namespace GL
{
	struct VertexArrayObjectNamespace
	{
		static GLuint gen();
		static void del(GLuint name);
	};

	typedef unique_name<VertexArrayObjectNamespace> VertexArray;
}

#endif  // INCLUDED_GL_VERTEX_ARRAY
