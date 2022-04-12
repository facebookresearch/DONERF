


#include "error.h"
#include "vertex_array.h"


namespace GL
{
	GLuint VertexArrayObjectNamespace::gen()
	{
		GLuint id;
		glGenVertexArrays(1, &id);
		checkError();
		return id;
	}

	void VertexArrayObjectNamespace::del(GLuint name)
	{
		glDeleteVertexArrays(1, &name);
		checkError();
	}
}
