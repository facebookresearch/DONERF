


#ifndef INCLUDED_GL_FRAMEBUFFER
#define INCLUDED_GL_FRAMEBUFFER

#pragma once

#include <GL/gl.h>

#include "unique_name.h"


namespace GL
{
	struct FramebufferObjectNamespace
	{
		static GLuint gen();
		static void del(GLuint name);
	};

	struct RenderbufferObjectNamespace
	{
		static GLuint gen();
		static void del(GLuint name);
	};

	typedef unique_name<FramebufferObjectNamespace> Framebuffer;
	typedef unique_name<RenderbufferObjectNamespace> Renderbuffer;

	Renderbuffer createRenderbuffer(GLsizei width, GLsizei height, GLenum format);
	Renderbuffer createRenderbuffer(GLsizei width, GLsizei height, GLenum format, GLsizei samples);
}

#endif  // INCLUDED_GL_FRAMEBUFFER
