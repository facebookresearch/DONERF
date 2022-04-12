


#ifndef INCLUDED_GL_TEXTURE
#define INCLUDED_GL_TEXTURE

#pragma once

#include <utility>

#include <GL/gl.h>

#include "error.h"


namespace GL
{
	class Texture
	{
	private:
		Texture(const Texture&);
		Texture& operator =(const Texture&);
		GLuint id;
		static GLuint create();
	protected:
		Texture();

		Texture(GLuint id)
			: id(id)
		{
		}

		Texture(Texture&& tex)
			: id(tex.id)
		{
			tex.id = 0;
		}

		~Texture();

		Texture& operator =(Texture&& tex)
		{
			using std::swap;
			swap(id, tex.id);
			return *this;
		}

		void swap(Texture& t)
		{
			using std::swap;
			swap(id, t.id);
		}

	public:
		operator GLuint() const { return id; }
	};

	class Texture1D : public Texture
	{
	private:
		Texture1D(const Texture1D&);
		Texture1D& operator =(const Texture1D&);
		using Texture::swap;
	public:
		Texture1D();
		Texture1D(GLuint id);
		Texture1D(GLsizei width, GLint levels, GLenum format);
		Texture1D(Texture1D&& tex)
			: Texture(std::forward<Texture>(tex))
		{
		}

		Texture1D& operator =(Texture1D&& tex)
		{
			Texture::operator =(std::forward<Texture>(tex));
			return *this;
		}

		void swap(Texture1D& t)
		{
			swap(static_cast<Texture&>(t));
		}
	};

	class Texture2D : public Texture
	{
	private:
		Texture2D(const Texture2D&);
		Texture2D& operator =(const Texture2D&);
		using Texture::swap;
	public:
		Texture2D();
		Texture2D(GLuint id);
		Texture2D(GLsizei width, GLsizei height, GLint levels, GLenum format);
		Texture2D(Texture2D&& tex)
			: Texture(std::forward<Texture>(tex))
		{
		}

		Texture2D& operator =(Texture2D&& tex)
		{
			Texture::operator =(std::forward<Texture>(tex));
			return *this;
		}

		void swap(Texture2D& t)
		{
			swap(static_cast<Texture&>(t));
		}
	};

	class Texture2DArray : public Texture
	{
	private:
		Texture2DArray(const Texture2DArray&);
		Texture2DArray& operator =(const Texture2DArray&);

		using Texture::swap;
	public:
		Texture2DArray();
		Texture2DArray(GLuint id);
		Texture2DArray(GLsizei width, GLsizei height, GLint slices, GLint levels, GLenum format);
		Texture2DArray(Texture2DArray&& tex)
			: Texture(std::forward<Texture>(tex))
		{
		}

		Texture2DArray& operator =(Texture2DArray&& tex)
		{
			Texture::operator =(std::forward<Texture>(tex));
			return *this;
		}

		void swap(Texture2DArray& t)
		{
			swap(static_cast<Texture&>(t));
		}
	};

	class Texture2DMS : public Texture
	{
	private:
		Texture2DMS(const Texture2DMS&);
		Texture2DMS& operator =(const Texture2DMS&);
		using Texture::swap;
	public:
		Texture2DMS();
		Texture2DMS(GLuint id);
		Texture2DMS(GLsizei width, GLsizei height, GLenum format, GLsizei samples);
		Texture2DMS(Texture2DMS&& tex)
			: Texture(std::forward<Texture>(tex))
		{
		}

		Texture2DMS& operator =(Texture2DMS&& tex)
		{
			Texture::operator =(std::forward<Texture>(tex));
			return *this;
		}

		void swap(Texture2DMS& t)
		{
			swap(static_cast<Texture&>(t));
		}
	};


	class Texture3D : public Texture
	{
	private:
		Texture3D(const Texture3D&);
		Texture3D& operator =(const Texture3D&);
		using Texture::swap;
	public:
		Texture3D();
		Texture3D(GLuint id);
		Texture3D(GLsizei width, GLsizei height, GLsizei depth, GLint levels, GLenum format);
		Texture3D(Texture3D&& tex)
			: Texture(std::forward<Texture>(tex))
		{
		}

		Texture3D& operator =(Texture3D&& tex)
		{
			Texture::operator =(std::forward<Texture>(tex));
			return *this;
		}

		void swap(Texture3D& t)
		{
			swap(static_cast<Texture&>(t));
		}
	};

	class TextureCube : public Texture
	{
	private:
		TextureCube(const TextureCube&);
		TextureCube& operator =(const TextureCube&);
		using Texture::swap;
	public:
		TextureCube();
		TextureCube(GLuint id);
		TextureCube(GLsizei width, GLint levels, GLenum format);
		TextureCube(TextureCube&& tex)
			: Texture(std::forward<Texture>(tex))
		{
		}

		TextureCube& operator =(TextureCube&& tex)
		{
			Texture::operator =(std::forward<Texture>(tex));
			return *this;
		}

		void swap(TextureCube& t)
		{
			swap(static_cast<Texture&>(t));
		}
	};

	class SamplerState
	{
	private:
		SamplerState(const Texture&);
		SamplerState& operator =(const SamplerState&);

		GLuint id;
		static GLuint create();
	public:
		SamplerState();
		~SamplerState();

		void swap(SamplerState& s)
		{
			using std::swap;
			swap(id, s.id);
		}

		operator GLuint() const { return id; }
	};
}

namespace std
{
	template <>
	inline void swap<GL::Texture1D>(GL::Texture1D& a, GL::Texture1D& b)
	{
		a.swap(b);
	}

	template <>
	inline void swap<GL::Texture2D>(GL::Texture2D& a, GL::Texture2D& b)
	{
		a.swap(b);
	}

	template <>
	inline void swap<GL::Texture2DMS>(GL::Texture2DMS& a, GL::Texture2DMS& b)
	{
		a.swap(b);
	}

	template <>
	inline void swap<GL::Texture3D>(GL::Texture3D& a, GL::Texture3D& b)
	{
		a.swap(b);
	}

	template <>
	inline void swap<GL::TextureCube>(GL::TextureCube& a, GL::TextureCube& b)
	{
		a.swap(b);
	}

	template <>
	inline void swap<GL::SamplerState>(GL::SamplerState& a, GL::SamplerState& b)
	{
		a.swap(b);
	}
}

#endif	// INCLUDED_GL_TEXTURE
