


#ifndef INCLUDED_GL_TRANSFORM_FEEDBACK
#define INCLUDED_GL_TRANSFORM_FEEDBACK

#pragma once

#include <GL/gl.h>

#include "unique_name.h"


namespace GL
{
	struct TransformFeedbackObjectNamespace
	{
		static GLuint gen();
		static void del(GLuint name);
	};

	typedef unique_name<TransformFeedbackObjectNamespace> TransformFeedback;
}

#endif  // INCLUDED_GL_TRANSFORM_FEEDBACK
