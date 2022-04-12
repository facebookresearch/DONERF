


#include "error.h"
#include "texture.h"


namespace GL
{
	GLuint TextureObjectNamespace::gen()
	{
		GLuint id;
		glGenTextures(1, &id);
		checkError();
		return id;
	}

	void TextureObjectNamespace::del(GLuint name)
	{
		glDeleteTextures(1, &name);
		checkError();
	}

	
	Texture createTexture2D(GLsizei width, GLsizei height, GLint levels, GLenum format)
	{
		Texture tex;
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexStorage2D(GL_TEXTURE_2D, levels, format, width, height);
		checkError();
		return tex;
	}

	Texture createTexture2DArray(GLsizei width, GLsizei height, GLint slices, GLint levels, GLenum format)
	{
		Texture tex;
		glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
		glTexStorage3D(GL_TEXTURE_2D_ARRAY, levels, format, width, height, slices);
		checkError();
		return tex;
	}

	Texture createTexture2DMS(GLsizei width, GLsizei height, GLenum format, GLsizei samples)
	{
		Texture tex;
		glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);
		glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, format, width, height, GL_FALSE);
		checkError();
		return tex;
	}

	Texture createTexture3D(GLsizei width, GLsizei height, GLsizei depth, GLint levels, GLenum format)
	{
		Texture tex;
		glBindTexture(GL_TEXTURE_3D, tex);
		glTexStorage3D(GL_TEXTURE_3D, levels, format, width, height, depth);
		checkError();
		return tex;
	}

	Texture createTextureCube(GLsizei width, GLint levels, GLenum format)
	{
		Texture tex;
		glBindTexture(GL_TEXTURE_CUBE_MAP, tex);
		glTexStorage2D(GL_TEXTURE_CUBE_MAP, levels, format, width, width);
		checkError();
		return tex;
	}


	GLuint SamplerObjectNamespace::gen()
	{
		GLuint id;
		glGenSamplers(1, &id);
		checkError();
		return id;
	}

	void SamplerObjectNamespace::del(GLuint name)
	{
		glDeleteSamplers(1, &name);
		checkError();
	}
}
