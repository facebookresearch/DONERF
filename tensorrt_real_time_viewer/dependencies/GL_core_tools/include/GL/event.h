


#ifndef INCLUDED_GL_EVENT
#define INCLUDED_GL_EVENT

#pragma once

#include <GL/gl.h>

#include "unique_name.h"


namespace GL
{
	struct QueryObjectNamespace
	{
		static GLuint gen();
		static void del(GLuint name);
	};

	typedef unique_name<QueryObjectNamespace> Query;
}

#endif  // INCLUDED_GL_EVENT
