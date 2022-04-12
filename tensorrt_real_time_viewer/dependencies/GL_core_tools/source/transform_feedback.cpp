


#include "error.h"
#include "transform_feedback.h"


namespace GL
{
	GLuint TransformFeedbackObjectNamespace::gen()
	{
		GLuint id;
		glGenTransformFeedbacks(1, &id);
		checkError();
		return id;
	}

	void TransformFeedbackObjectNamespace::del(GLuint name)
	{
		glDeleteTransformFeedbacks(1, &name);
		checkError();
	}
}
