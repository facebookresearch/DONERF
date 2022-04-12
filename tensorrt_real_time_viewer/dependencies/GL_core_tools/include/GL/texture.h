


#ifndef INCLUDED_GL_TEXTURE
#define INCLUDED_GL_TEXTURE

#pragma once

#include <GL/gl.h>

#include "unique_name.h"


namespace GL
{
	struct TextureObjectNamespace
	{
		static GLuint gen();
		static void del(GLuint name);
	};

	struct SamplerObjectNamespace
	{
		static GLuint gen();
		static void del(GLuint name);
	};

	typedef unique_name<TextureObjectNamespace> Texture;
	typedef unique_name<SamplerObjectNamespace> Sampler;

	Texture createTexture1D(GLsizei width, GLint levels, GLenum format);
	
	Texture createTexture2D(GLsizei width, GLsizei height, GLint levels, GLenum format);
	
	Texture createTexture2DArray(GLsizei width, GLsizei height, GLint slices, GLint levels, GLenum format);
	
	Texture createTexture2DMS(GLsizei width, GLsizei height, GLenum format, GLsizei samples);
	
	Texture createTexture3D(GLsizei width, GLsizei height, GLsizei depth, GLint levels, GLenum format);
	
	Texture createTextureCube(GLsizei width, GLint levels, GLenum format);
	
}

#endif  // INCLUDED_GL_TEXTURE
