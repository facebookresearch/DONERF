


#include "error.h"
#include "buffer.h"


namespace GL
{
	GLuint BufferObjectNamespace::gen()
	{
		GLuint id;
		glGenBuffers(1, &id);
		checkError();
		return id;
	}

	void BufferObjectNamespace::del(GLuint name)
	{
		glDeleteBuffers(1, &name);
		checkError();
	}
}
