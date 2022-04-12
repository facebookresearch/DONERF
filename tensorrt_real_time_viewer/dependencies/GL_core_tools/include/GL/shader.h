


#ifndef INCLUDED_GL_SHADER
#define INCLUDED_GL_SHADER

#pragma once

#include <string>
#include <stdexcept>

#include <GL/gl.h>

#include "unique_name.h"


namespace GL
{
	class compile_error : public std::runtime_error
	{
	public:
		compile_error(std::string log);
	};

	class link_error : public std::runtime_error
	{
	public:
		link_error(std::string log);
	};


	GLuint createShader(GLenum type);
	void deleteShader(GLuint name);

	template <GLenum ShaderType>
	struct ShaderObjectNamespace
	{
		static GLuint gen()
		{
			return createShader(ShaderType);
		}

		static void del(GLuint name)
		{
			deleteShader(name);
		}
	};

	template <GLenum ShaderType>
	using Shader = unique_name<ShaderObjectNamespace<ShaderType>>;

	typedef Shader<GL_VERTEX_SHADER> VertexShader;
	typedef Shader<GL_TESS_CONTROL_SHADER> TessellationControlShader;
	typedef Shader<GL_TESS_EVALUATION_SHADER> TessellationEvaluationShader;
	typedef Shader<GL_GEOMETRY_SHADER> GeometryShader;
	typedef Shader<GL_FRAGMENT_SHADER> FragmentShader;
	typedef Shader<GL_COMPUTE_SHADER> ComputeShader;

	bool getShaderCompileStatus(GLuint shader);
	std::string getShaderInfoLog(GLuint shader);

	VertexShader compileVertexShader(const char* source);
	TessellationControlShader compileTessellationControlShader(const char* source);
	TessellationEvaluationShader compileTessellationEvaluationShader(const char* source);
	GeometryShader compileGeometryShader(const char* source);
	FragmentShader compileFragmentShader(const char* source);
	ComputeShader compileComputeShader(const char* source);


	struct ProgramObjectNamespace
	{
		static GLuint gen();
		static void del(GLuint name);
	};

	typedef unique_name<ProgramObjectNamespace> Program;


	bool getProgramLinkStatus(GLuint program);
	bool getProgramValidationStatus(GLuint program);
	std::string getProgramInfoLog(GLuint program);

	void linkProgram(GLuint program);


	struct ProgramPipelineObjectNamespace
	{
		static GLuint gen();
		static void del(GLuint name);
	};

	typedef unique_name<ProgramObjectNamespace> ProgramPipeline;
}

#endif  // INCLUDED_GL_SHADER
