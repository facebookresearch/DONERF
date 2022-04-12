


#include "error.h"
#include "event.h"


namespace GL
{
	GLuint QueryObjectNamespace::gen()
	{
		GLuint name;
		glGenQueries(1, &name);
		checkError();
		return name;
	}

	void QueryObjectNamespace::del(GLuint name)
	{
		glDeleteQueries(1, &name);
		checkError();
	}
}
