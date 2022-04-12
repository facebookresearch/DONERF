


#include "error.h"
#include "framebuffer.h"


namespace GL
{
	GLuint FramebufferObjectNamespace::gen()
	{
		GLuint id;
		glGenFramebuffers(1, &id);
		checkError();
		return id;
	}

	void FramebufferObjectNamespace::del(GLuint name)
	{
		glDeleteFramebuffers(1, &name);
		checkError();
	}


	GLuint RenderbufferObjectNamespace::gen()
	{
		GLuint id;
		glGenRenderbuffers(1, &id);
		checkError();
		return id;
	}

	void RenderbufferObjectNamespace::del(GLuint name)
	{
		glDeleteRenderbuffers(1, &name);
		checkError();
	}
	

	Renderbuffer createRenderbuffer(GLsizei width, GLsizei height, GLenum format)
	{
		Renderbuffer renderbuffer;
		glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, format, width, height);
		checkError();
		return renderbuffer;
	}

	Renderbuffer createRenderbuffer(GLsizei width, GLsizei height, GLenum format, GLsizei samples)
	{
		Renderbuffer renderbuffer;
		glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
		glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, format, width, height);
		checkError();
		return renderbuffer;
	}
}
