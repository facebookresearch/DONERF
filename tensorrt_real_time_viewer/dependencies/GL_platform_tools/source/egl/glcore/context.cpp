


#include <stdexcept>

#include "context.h"


namespace
{
	thread_local const glcoreContext* context = nullptr;
}

extern "C"
{
	glcoreContext::glcoreContext()
	{
		CullFace = reinterpret_cast<decltype(glCullFace)*>(eglGetProcAddress("glCullFace"));
		FrontFace = reinterpret_cast<decltype(glFrontFace)*>(eglGetProcAddress("glFrontFace"));
		Hint = reinterpret_cast<decltype(glHint)*>(eglGetProcAddress("glHint"));
		LineWidth = reinterpret_cast<decltype(glLineWidth)*>(eglGetProcAddress("glLineWidth"));
		PointSize = reinterpret_cast<decltype(glPointSize)*>(eglGetProcAddress("glPointSize"));
		PolygonMode = reinterpret_cast<decltype(glPolygonMode)*>(eglGetProcAddress("glPolygonMode"));
		Scissor = reinterpret_cast<decltype(glScissor)*>(eglGetProcAddress("glScissor"));
		TexParameterf = reinterpret_cast<decltype(glTexParameterf)*>(eglGetProcAddress("glTexParameterf"));
		TexParameterfv = reinterpret_cast<decltype(glTexParameterfv)*>(eglGetProcAddress("glTexParameterfv"));
		TexParameteri = reinterpret_cast<decltype(glTexParameteri)*>(eglGetProcAddress("glTexParameteri"));
		TexParameteriv = reinterpret_cast<decltype(glTexParameteriv)*>(eglGetProcAddress("glTexParameteriv"));
		TexImage1D = reinterpret_cast<decltype(glTexImage1D)*>(eglGetProcAddress("glTexImage1D"));
		TexImage2D = reinterpret_cast<decltype(glTexImage2D)*>(eglGetProcAddress("glTexImage2D"));
		DrawBuffer = reinterpret_cast<decltype(glDrawBuffer)*>(eglGetProcAddress("glDrawBuffer"));
		Clear = reinterpret_cast<decltype(glClear)*>(eglGetProcAddress("glClear"));
		ClearColor = reinterpret_cast<decltype(glClearColor)*>(eglGetProcAddress("glClearColor"));
		ClearStencil = reinterpret_cast<decltype(glClearStencil)*>(eglGetProcAddress("glClearStencil"));
		ClearDepth = reinterpret_cast<decltype(glClearDepth)*>(eglGetProcAddress("glClearDepth"));
		StencilMask = reinterpret_cast<decltype(glStencilMask)*>(eglGetProcAddress("glStencilMask"));
		ColorMask = reinterpret_cast<decltype(glColorMask)*>(eglGetProcAddress("glColorMask"));
		DepthMask = reinterpret_cast<decltype(glDepthMask)*>(eglGetProcAddress("glDepthMask"));
		Disable = reinterpret_cast<decltype(glDisable)*>(eglGetProcAddress("glDisable"));
		Enable = reinterpret_cast<decltype(glEnable)*>(eglGetProcAddress("glEnable"));
		Finish = reinterpret_cast<decltype(glFinish)*>(eglGetProcAddress("glFinish"));
		Flush = reinterpret_cast<decltype(glFlush)*>(eglGetProcAddress("glFlush"));
		BlendFunc = reinterpret_cast<decltype(glBlendFunc)*>(eglGetProcAddress("glBlendFunc"));
		LogicOp = reinterpret_cast<decltype(glLogicOp)*>(eglGetProcAddress("glLogicOp"));
		StencilFunc = reinterpret_cast<decltype(glStencilFunc)*>(eglGetProcAddress("glStencilFunc"));
		StencilOp = reinterpret_cast<decltype(glStencilOp)*>(eglGetProcAddress("glStencilOp"));
		DepthFunc = reinterpret_cast<decltype(glDepthFunc)*>(eglGetProcAddress("glDepthFunc"));
		PixelStoref = reinterpret_cast<decltype(glPixelStoref)*>(eglGetProcAddress("glPixelStoref"));
		PixelStorei = reinterpret_cast<decltype(glPixelStorei)*>(eglGetProcAddress("glPixelStorei"));
		ReadBuffer = reinterpret_cast<decltype(glReadBuffer)*>(eglGetProcAddress("glReadBuffer"));
		ReadPixels = reinterpret_cast<decltype(glReadPixels)*>(eglGetProcAddress("glReadPixels"));
		GetBooleanv = reinterpret_cast<decltype(glGetBooleanv)*>(eglGetProcAddress("glGetBooleanv"));
		GetDoublev = reinterpret_cast<decltype(glGetDoublev)*>(eglGetProcAddress("glGetDoublev"));
		GetError = reinterpret_cast<decltype(glGetError)*>(eglGetProcAddress("glGetError"));
		GetFloatv = reinterpret_cast<decltype(glGetFloatv)*>(eglGetProcAddress("glGetFloatv"));
		GetIntegerv = reinterpret_cast<decltype(glGetIntegerv)*>(eglGetProcAddress("glGetIntegerv"));
		GetString = reinterpret_cast<decltype(glGetString)*>(eglGetProcAddress("glGetString"));
		GetTexImage = reinterpret_cast<decltype(glGetTexImage)*>(eglGetProcAddress("glGetTexImage"));
		GetTexParameterfv = reinterpret_cast<decltype(glGetTexParameterfv)*>(eglGetProcAddress("glGetTexParameterfv"));
		GetTexParameteriv = reinterpret_cast<decltype(glGetTexParameteriv)*>(eglGetProcAddress("glGetTexParameteriv"));
		GetTexLevelParameterfv = reinterpret_cast<decltype(glGetTexLevelParameterfv)*>(eglGetProcAddress("glGetTexLevelParameterfv"));
		GetTexLevelParameteriv = reinterpret_cast<decltype(glGetTexLevelParameteriv)*>(eglGetProcAddress("glGetTexLevelParameteriv"));
		IsEnabled = reinterpret_cast<decltype(glIsEnabled)*>(eglGetProcAddress("glIsEnabled"));
		DepthRange = reinterpret_cast<decltype(glDepthRange)*>(eglGetProcAddress("glDepthRange"));
		Viewport = reinterpret_cast<decltype(glViewport)*>(eglGetProcAddress("glViewport"));
		DrawArrays = reinterpret_cast<decltype(glDrawArrays)*>(eglGetProcAddress("glDrawArrays"));
		DrawElements = reinterpret_cast<decltype(glDrawElements)*>(eglGetProcAddress("glDrawElements"));
		PolygonOffset = reinterpret_cast<decltype(glPolygonOffset)*>(eglGetProcAddress("glPolygonOffset"));
		CopyTexImage1D = reinterpret_cast<decltype(glCopyTexImage1D)*>(eglGetProcAddress("glCopyTexImage1D"));
		CopyTexImage2D = reinterpret_cast<decltype(glCopyTexImage2D)*>(eglGetProcAddress("glCopyTexImage2D"));
		CopyTexSubImage1D = reinterpret_cast<decltype(glCopyTexSubImage1D)*>(eglGetProcAddress("glCopyTexSubImage1D"));
		CopyTexSubImage2D = reinterpret_cast<decltype(glCopyTexSubImage2D)*>(eglGetProcAddress("glCopyTexSubImage2D"));
		TexSubImage1D = reinterpret_cast<decltype(glTexSubImage1D)*>(eglGetProcAddress("glTexSubImage1D"));
		TexSubImage2D = reinterpret_cast<decltype(glTexSubImage2D)*>(eglGetProcAddress("glTexSubImage2D"));
		BindTexture = reinterpret_cast<decltype(glBindTexture)*>(eglGetProcAddress("glBindTexture"));
		DeleteTextures = reinterpret_cast<decltype(glDeleteTextures)*>(eglGetProcAddress("glDeleteTextures"));
		GenTextures = reinterpret_cast<decltype(glGenTextures)*>(eglGetProcAddress("glGenTextures"));
		IsTexture = reinterpret_cast<decltype(glIsTexture)*>(eglGetProcAddress("glIsTexture"));
		DrawRangeElements = reinterpret_cast<decltype(glDrawRangeElements)*>(eglGetProcAddress("glDrawRangeElements"));
		TexImage3D = reinterpret_cast<decltype(glTexImage3D)*>(eglGetProcAddress("glTexImage3D"));
		TexSubImage3D = reinterpret_cast<decltype(glTexSubImage3D)*>(eglGetProcAddress("glTexSubImage3D"));
		CopyTexSubImage3D = reinterpret_cast<decltype(glCopyTexSubImage3D)*>(eglGetProcAddress("glCopyTexSubImage3D"));
		ActiveTexture = reinterpret_cast<decltype(glActiveTexture)*>(eglGetProcAddress("glActiveTexture"));
		SampleCoverage = reinterpret_cast<decltype(glSampleCoverage)*>(eglGetProcAddress("glSampleCoverage"));
		CompressedTexImage3D = reinterpret_cast<decltype(glCompressedTexImage3D)*>(eglGetProcAddress("glCompressedTexImage3D"));
		CompressedTexImage2D = reinterpret_cast<decltype(glCompressedTexImage2D)*>(eglGetProcAddress("glCompressedTexImage2D"));
		CompressedTexImage1D = reinterpret_cast<decltype(glCompressedTexImage1D)*>(eglGetProcAddress("glCompressedTexImage1D"));
		CompressedTexSubImage3D = reinterpret_cast<decltype(glCompressedTexSubImage3D)*>(eglGetProcAddress("glCompressedTexSubImage3D"));
		CompressedTexSubImage2D = reinterpret_cast<decltype(glCompressedTexSubImage2D)*>(eglGetProcAddress("glCompressedTexSubImage2D"));
		CompressedTexSubImage1D = reinterpret_cast<decltype(glCompressedTexSubImage1D)*>(eglGetProcAddress("glCompressedTexSubImage1D"));
		GetCompressedTexImage = reinterpret_cast<decltype(glGetCompressedTexImage)*>(eglGetProcAddress("glGetCompressedTexImage"));
		BlendFuncSeparate = reinterpret_cast<decltype(glBlendFuncSeparate)*>(eglGetProcAddress("glBlendFuncSeparate"));
		MultiDrawArrays = reinterpret_cast<decltype(glMultiDrawArrays)*>(eglGetProcAddress("glMultiDrawArrays"));
		MultiDrawElements = reinterpret_cast<decltype(glMultiDrawElements)*>(eglGetProcAddress("glMultiDrawElements"));
		PointParameterf = reinterpret_cast<decltype(glPointParameterf)*>(eglGetProcAddress("glPointParameterf"));
		PointParameterfv = reinterpret_cast<decltype(glPointParameterfv)*>(eglGetProcAddress("glPointParameterfv"));
		PointParameteri = reinterpret_cast<decltype(glPointParameteri)*>(eglGetProcAddress("glPointParameteri"));
		PointParameteriv = reinterpret_cast<decltype(glPointParameteriv)*>(eglGetProcAddress("glPointParameteriv"));
		BlendColor = reinterpret_cast<decltype(glBlendColor)*>(eglGetProcAddress("glBlendColor"));
		BlendEquation = reinterpret_cast<decltype(glBlendEquation)*>(eglGetProcAddress("glBlendEquation"));
		GenQueries = reinterpret_cast<decltype(glGenQueries)*>(eglGetProcAddress("glGenQueries"));
		DeleteQueries = reinterpret_cast<decltype(glDeleteQueries)*>(eglGetProcAddress("glDeleteQueries"));
		IsQuery = reinterpret_cast<decltype(glIsQuery)*>(eglGetProcAddress("glIsQuery"));
		BeginQuery = reinterpret_cast<decltype(glBeginQuery)*>(eglGetProcAddress("glBeginQuery"));
		EndQuery = reinterpret_cast<decltype(glEndQuery)*>(eglGetProcAddress("glEndQuery"));
		GetQueryiv = reinterpret_cast<decltype(glGetQueryiv)*>(eglGetProcAddress("glGetQueryiv"));
		GetQueryObjectiv = reinterpret_cast<decltype(glGetQueryObjectiv)*>(eglGetProcAddress("glGetQueryObjectiv"));
		GetQueryObjectuiv = reinterpret_cast<decltype(glGetQueryObjectuiv)*>(eglGetProcAddress("glGetQueryObjectuiv"));
		BindBuffer = reinterpret_cast<decltype(glBindBuffer)*>(eglGetProcAddress("glBindBuffer"));
		DeleteBuffers = reinterpret_cast<decltype(glDeleteBuffers)*>(eglGetProcAddress("glDeleteBuffers"));
		GenBuffers = reinterpret_cast<decltype(glGenBuffers)*>(eglGetProcAddress("glGenBuffers"));
		IsBuffer = reinterpret_cast<decltype(glIsBuffer)*>(eglGetProcAddress("glIsBuffer"));
		BufferData = reinterpret_cast<decltype(glBufferData)*>(eglGetProcAddress("glBufferData"));
		BufferSubData = reinterpret_cast<decltype(glBufferSubData)*>(eglGetProcAddress("glBufferSubData"));
		GetBufferSubData = reinterpret_cast<decltype(glGetBufferSubData)*>(eglGetProcAddress("glGetBufferSubData"));
		MapBuffer = reinterpret_cast<decltype(glMapBuffer)*>(eglGetProcAddress("glMapBuffer"));
		UnmapBuffer = reinterpret_cast<decltype(glUnmapBuffer)*>(eglGetProcAddress("glUnmapBuffer"));
		GetBufferParameteriv = reinterpret_cast<decltype(glGetBufferParameteriv)*>(eglGetProcAddress("glGetBufferParameteriv"));
		GetBufferPointerv = reinterpret_cast<decltype(glGetBufferPointerv)*>(eglGetProcAddress("glGetBufferPointerv"));
		BlendEquationSeparate = reinterpret_cast<decltype(glBlendEquationSeparate)*>(eglGetProcAddress("glBlendEquationSeparate"));
		DrawBuffers = reinterpret_cast<decltype(glDrawBuffers)*>(eglGetProcAddress("glDrawBuffers"));
		StencilOpSeparate = reinterpret_cast<decltype(glStencilOpSeparate)*>(eglGetProcAddress("glStencilOpSeparate"));
		StencilFuncSeparate = reinterpret_cast<decltype(glStencilFuncSeparate)*>(eglGetProcAddress("glStencilFuncSeparate"));
		StencilMaskSeparate = reinterpret_cast<decltype(glStencilMaskSeparate)*>(eglGetProcAddress("glStencilMaskSeparate"));
		AttachShader = reinterpret_cast<decltype(glAttachShader)*>(eglGetProcAddress("glAttachShader"));
		BindAttribLocation = reinterpret_cast<decltype(glBindAttribLocation)*>(eglGetProcAddress("glBindAttribLocation"));
		CompileShader = reinterpret_cast<decltype(glCompileShader)*>(eglGetProcAddress("glCompileShader"));
		CreateProgram = reinterpret_cast<decltype(glCreateProgram)*>(eglGetProcAddress("glCreateProgram"));
		CreateShader = reinterpret_cast<decltype(glCreateShader)*>(eglGetProcAddress("glCreateShader"));
		DeleteProgram = reinterpret_cast<decltype(glDeleteProgram)*>(eglGetProcAddress("glDeleteProgram"));
		DeleteShader = reinterpret_cast<decltype(glDeleteShader)*>(eglGetProcAddress("glDeleteShader"));
		DetachShader = reinterpret_cast<decltype(glDetachShader)*>(eglGetProcAddress("glDetachShader"));
		DisableVertexAttribArray = reinterpret_cast<decltype(glDisableVertexAttribArray)*>(eglGetProcAddress("glDisableVertexAttribArray"));
		EnableVertexAttribArray = reinterpret_cast<decltype(glEnableVertexAttribArray)*>(eglGetProcAddress("glEnableVertexAttribArray"));
		GetActiveAttrib = reinterpret_cast<decltype(glGetActiveAttrib)*>(eglGetProcAddress("glGetActiveAttrib"));
		GetActiveUniform = reinterpret_cast<decltype(glGetActiveUniform)*>(eglGetProcAddress("glGetActiveUniform"));
		GetAttachedShaders = reinterpret_cast<decltype(glGetAttachedShaders)*>(eglGetProcAddress("glGetAttachedShaders"));
		GetAttribLocation = reinterpret_cast<decltype(glGetAttribLocation)*>(eglGetProcAddress("glGetAttribLocation"));
		GetProgramiv = reinterpret_cast<decltype(glGetProgramiv)*>(eglGetProcAddress("glGetProgramiv"));
		GetProgramInfoLog = reinterpret_cast<decltype(glGetProgramInfoLog)*>(eglGetProcAddress("glGetProgramInfoLog"));
		GetShaderiv = reinterpret_cast<decltype(glGetShaderiv)*>(eglGetProcAddress("glGetShaderiv"));
		GetShaderInfoLog = reinterpret_cast<decltype(glGetShaderInfoLog)*>(eglGetProcAddress("glGetShaderInfoLog"));
		GetShaderSource = reinterpret_cast<decltype(glGetShaderSource)*>(eglGetProcAddress("glGetShaderSource"));
		GetUniformLocation = reinterpret_cast<decltype(glGetUniformLocation)*>(eglGetProcAddress("glGetUniformLocation"));
		GetUniformfv = reinterpret_cast<decltype(glGetUniformfv)*>(eglGetProcAddress("glGetUniformfv"));
		GetUniformiv = reinterpret_cast<decltype(glGetUniformiv)*>(eglGetProcAddress("glGetUniformiv"));
		GetVertexAttribdv = reinterpret_cast<decltype(glGetVertexAttribdv)*>(eglGetProcAddress("glGetVertexAttribdv"));
		GetVertexAttribfv = reinterpret_cast<decltype(glGetVertexAttribfv)*>(eglGetProcAddress("glGetVertexAttribfv"));
		GetVertexAttribiv = reinterpret_cast<decltype(glGetVertexAttribiv)*>(eglGetProcAddress("glGetVertexAttribiv"));
		GetVertexAttribPointerv = reinterpret_cast<decltype(glGetVertexAttribPointerv)*>(eglGetProcAddress("glGetVertexAttribPointerv"));
		IsProgram = reinterpret_cast<decltype(glIsProgram)*>(eglGetProcAddress("glIsProgram"));
		IsShader = reinterpret_cast<decltype(glIsShader)*>(eglGetProcAddress("glIsShader"));
		LinkProgram = reinterpret_cast<decltype(glLinkProgram)*>(eglGetProcAddress("glLinkProgram"));
		ShaderSource = reinterpret_cast<decltype(glShaderSource)*>(eglGetProcAddress("glShaderSource"));
		UseProgram = reinterpret_cast<decltype(glUseProgram)*>(eglGetProcAddress("glUseProgram"));
		Uniform1f = reinterpret_cast<decltype(glUniform1f)*>(eglGetProcAddress("glUniform1f"));
		Uniform2f = reinterpret_cast<decltype(glUniform2f)*>(eglGetProcAddress("glUniform2f"));
		Uniform3f = reinterpret_cast<decltype(glUniform3f)*>(eglGetProcAddress("glUniform3f"));
		Uniform4f = reinterpret_cast<decltype(glUniform4f)*>(eglGetProcAddress("glUniform4f"));
		Uniform1i = reinterpret_cast<decltype(glUniform1i)*>(eglGetProcAddress("glUniform1i"));
		Uniform2i = reinterpret_cast<decltype(glUniform2i)*>(eglGetProcAddress("glUniform2i"));
		Uniform3i = reinterpret_cast<decltype(glUniform3i)*>(eglGetProcAddress("glUniform3i"));
		Uniform4i = reinterpret_cast<decltype(glUniform4i)*>(eglGetProcAddress("glUniform4i"));
		Uniform1fv = reinterpret_cast<decltype(glUniform1fv)*>(eglGetProcAddress("glUniform1fv"));
		Uniform2fv = reinterpret_cast<decltype(glUniform2fv)*>(eglGetProcAddress("glUniform2fv"));
		Uniform3fv = reinterpret_cast<decltype(glUniform3fv)*>(eglGetProcAddress("glUniform3fv"));
		Uniform4fv = reinterpret_cast<decltype(glUniform4fv)*>(eglGetProcAddress("glUniform4fv"));
		Uniform1iv = reinterpret_cast<decltype(glUniform1iv)*>(eglGetProcAddress("glUniform1iv"));
		Uniform2iv = reinterpret_cast<decltype(glUniform2iv)*>(eglGetProcAddress("glUniform2iv"));
		Uniform3iv = reinterpret_cast<decltype(glUniform3iv)*>(eglGetProcAddress("glUniform3iv"));
		Uniform4iv = reinterpret_cast<decltype(glUniform4iv)*>(eglGetProcAddress("glUniform4iv"));
		UniformMatrix2fv = reinterpret_cast<decltype(glUniformMatrix2fv)*>(eglGetProcAddress("glUniformMatrix2fv"));
		UniformMatrix3fv = reinterpret_cast<decltype(glUniformMatrix3fv)*>(eglGetProcAddress("glUniformMatrix3fv"));
		UniformMatrix4fv = reinterpret_cast<decltype(glUniformMatrix4fv)*>(eglGetProcAddress("glUniformMatrix4fv"));
		ValidateProgram = reinterpret_cast<decltype(glValidateProgram)*>(eglGetProcAddress("glValidateProgram"));
		VertexAttrib1d = reinterpret_cast<decltype(glVertexAttrib1d)*>(eglGetProcAddress("glVertexAttrib1d"));
		VertexAttrib1dv = reinterpret_cast<decltype(glVertexAttrib1dv)*>(eglGetProcAddress("glVertexAttrib1dv"));
		VertexAttrib1f = reinterpret_cast<decltype(glVertexAttrib1f)*>(eglGetProcAddress("glVertexAttrib1f"));
		VertexAttrib1fv = reinterpret_cast<decltype(glVertexAttrib1fv)*>(eglGetProcAddress("glVertexAttrib1fv"));
		VertexAttrib1s = reinterpret_cast<decltype(glVertexAttrib1s)*>(eglGetProcAddress("glVertexAttrib1s"));
		VertexAttrib1sv = reinterpret_cast<decltype(glVertexAttrib1sv)*>(eglGetProcAddress("glVertexAttrib1sv"));
		VertexAttrib2d = reinterpret_cast<decltype(glVertexAttrib2d)*>(eglGetProcAddress("glVertexAttrib2d"));
		VertexAttrib2dv = reinterpret_cast<decltype(glVertexAttrib2dv)*>(eglGetProcAddress("glVertexAttrib2dv"));
		VertexAttrib2f = reinterpret_cast<decltype(glVertexAttrib2f)*>(eglGetProcAddress("glVertexAttrib2f"));
		VertexAttrib2fv = reinterpret_cast<decltype(glVertexAttrib2fv)*>(eglGetProcAddress("glVertexAttrib2fv"));
		VertexAttrib2s = reinterpret_cast<decltype(glVertexAttrib2s)*>(eglGetProcAddress("glVertexAttrib2s"));
		VertexAttrib2sv = reinterpret_cast<decltype(glVertexAttrib2sv)*>(eglGetProcAddress("glVertexAttrib2sv"));
		VertexAttrib3d = reinterpret_cast<decltype(glVertexAttrib3d)*>(eglGetProcAddress("glVertexAttrib3d"));
		VertexAttrib3dv = reinterpret_cast<decltype(glVertexAttrib3dv)*>(eglGetProcAddress("glVertexAttrib3dv"));
		VertexAttrib3f = reinterpret_cast<decltype(glVertexAttrib3f)*>(eglGetProcAddress("glVertexAttrib3f"));
		VertexAttrib3fv = reinterpret_cast<decltype(glVertexAttrib3fv)*>(eglGetProcAddress("glVertexAttrib3fv"));
		VertexAttrib3s = reinterpret_cast<decltype(glVertexAttrib3s)*>(eglGetProcAddress("glVertexAttrib3s"));
		VertexAttrib3sv = reinterpret_cast<decltype(glVertexAttrib3sv)*>(eglGetProcAddress("glVertexAttrib3sv"));
		VertexAttrib4Nbv = reinterpret_cast<decltype(glVertexAttrib4Nbv)*>(eglGetProcAddress("glVertexAttrib4Nbv"));
		VertexAttrib4Niv = reinterpret_cast<decltype(glVertexAttrib4Niv)*>(eglGetProcAddress("glVertexAttrib4Niv"));
		VertexAttrib4Nsv = reinterpret_cast<decltype(glVertexAttrib4Nsv)*>(eglGetProcAddress("glVertexAttrib4Nsv"));
		VertexAttrib4Nub = reinterpret_cast<decltype(glVertexAttrib4Nub)*>(eglGetProcAddress("glVertexAttrib4Nub"));
		VertexAttrib4Nubv = reinterpret_cast<decltype(glVertexAttrib4Nubv)*>(eglGetProcAddress("glVertexAttrib4Nubv"));
		VertexAttrib4Nuiv = reinterpret_cast<decltype(glVertexAttrib4Nuiv)*>(eglGetProcAddress("glVertexAttrib4Nuiv"));
		VertexAttrib4Nusv = reinterpret_cast<decltype(glVertexAttrib4Nusv)*>(eglGetProcAddress("glVertexAttrib4Nusv"));
		VertexAttrib4bv = reinterpret_cast<decltype(glVertexAttrib4bv)*>(eglGetProcAddress("glVertexAttrib4bv"));
		VertexAttrib4d = reinterpret_cast<decltype(glVertexAttrib4d)*>(eglGetProcAddress("glVertexAttrib4d"));
		VertexAttrib4dv = reinterpret_cast<decltype(glVertexAttrib4dv)*>(eglGetProcAddress("glVertexAttrib4dv"));
		VertexAttrib4f = reinterpret_cast<decltype(glVertexAttrib4f)*>(eglGetProcAddress("glVertexAttrib4f"));
		VertexAttrib4fv = reinterpret_cast<decltype(glVertexAttrib4fv)*>(eglGetProcAddress("glVertexAttrib4fv"));
		VertexAttrib4iv = reinterpret_cast<decltype(glVertexAttrib4iv)*>(eglGetProcAddress("glVertexAttrib4iv"));
		VertexAttrib4s = reinterpret_cast<decltype(glVertexAttrib4s)*>(eglGetProcAddress("glVertexAttrib4s"));
		VertexAttrib4sv = reinterpret_cast<decltype(glVertexAttrib4sv)*>(eglGetProcAddress("glVertexAttrib4sv"));
		VertexAttrib4ubv = reinterpret_cast<decltype(glVertexAttrib4ubv)*>(eglGetProcAddress("glVertexAttrib4ubv"));
		VertexAttrib4uiv = reinterpret_cast<decltype(glVertexAttrib4uiv)*>(eglGetProcAddress("glVertexAttrib4uiv"));
		VertexAttrib4usv = reinterpret_cast<decltype(glVertexAttrib4usv)*>(eglGetProcAddress("glVertexAttrib4usv"));
		VertexAttribPointer = reinterpret_cast<decltype(glVertexAttribPointer)*>(eglGetProcAddress("glVertexAttribPointer"));
		UniformMatrix2x3fv = reinterpret_cast<decltype(glUniformMatrix2x3fv)*>(eglGetProcAddress("glUniformMatrix2x3fv"));
		UniformMatrix3x2fv = reinterpret_cast<decltype(glUniformMatrix3x2fv)*>(eglGetProcAddress("glUniformMatrix3x2fv"));
		UniformMatrix2x4fv = reinterpret_cast<decltype(glUniformMatrix2x4fv)*>(eglGetProcAddress("glUniformMatrix2x4fv"));
		UniformMatrix4x2fv = reinterpret_cast<decltype(glUniformMatrix4x2fv)*>(eglGetProcAddress("glUniformMatrix4x2fv"));
		UniformMatrix3x4fv = reinterpret_cast<decltype(glUniformMatrix3x4fv)*>(eglGetProcAddress("glUniformMatrix3x4fv"));
		UniformMatrix4x3fv = reinterpret_cast<decltype(glUniformMatrix4x3fv)*>(eglGetProcAddress("glUniformMatrix4x3fv"));
		ColorMaski = reinterpret_cast<decltype(glColorMaski)*>(eglGetProcAddress("glColorMaski"));
		GetBooleani_v = reinterpret_cast<decltype(glGetBooleani_v)*>(eglGetProcAddress("glGetBooleani_v"));
		GetIntegeri_v = reinterpret_cast<decltype(glGetIntegeri_v)*>(eglGetProcAddress("glGetIntegeri_v"));
		Enablei = reinterpret_cast<decltype(glEnablei)*>(eglGetProcAddress("glEnablei"));
		Disablei = reinterpret_cast<decltype(glDisablei)*>(eglGetProcAddress("glDisablei"));
		IsEnabledi = reinterpret_cast<decltype(glIsEnabledi)*>(eglGetProcAddress("glIsEnabledi"));
		BeginTransformFeedback = reinterpret_cast<decltype(glBeginTransformFeedback)*>(eglGetProcAddress("glBeginTransformFeedback"));
		EndTransformFeedback = reinterpret_cast<decltype(glEndTransformFeedback)*>(eglGetProcAddress("glEndTransformFeedback"));
		BindBufferRange = reinterpret_cast<decltype(glBindBufferRange)*>(eglGetProcAddress("glBindBufferRange"));
		BindBufferBase = reinterpret_cast<decltype(glBindBufferBase)*>(eglGetProcAddress("glBindBufferBase"));
		TransformFeedbackVaryings = reinterpret_cast<decltype(glTransformFeedbackVaryings)*>(eglGetProcAddress("glTransformFeedbackVaryings"));
		GetTransformFeedbackVarying = reinterpret_cast<decltype(glGetTransformFeedbackVarying)*>(eglGetProcAddress("glGetTransformFeedbackVarying"));
		ClampColor = reinterpret_cast<decltype(glClampColor)*>(eglGetProcAddress("glClampColor"));
		BeginConditionalRender = reinterpret_cast<decltype(glBeginConditionalRender)*>(eglGetProcAddress("glBeginConditionalRender"));
		EndConditionalRender = reinterpret_cast<decltype(glEndConditionalRender)*>(eglGetProcAddress("glEndConditionalRender"));
		VertexAttribIPointer = reinterpret_cast<decltype(glVertexAttribIPointer)*>(eglGetProcAddress("glVertexAttribIPointer"));
		GetVertexAttribIiv = reinterpret_cast<decltype(glGetVertexAttribIiv)*>(eglGetProcAddress("glGetVertexAttribIiv"));
		GetVertexAttribIuiv = reinterpret_cast<decltype(glGetVertexAttribIuiv)*>(eglGetProcAddress("glGetVertexAttribIuiv"));
		VertexAttribI1i = reinterpret_cast<decltype(glVertexAttribI1i)*>(eglGetProcAddress("glVertexAttribI1i"));
		VertexAttribI2i = reinterpret_cast<decltype(glVertexAttribI2i)*>(eglGetProcAddress("glVertexAttribI2i"));
		VertexAttribI3i = reinterpret_cast<decltype(glVertexAttribI3i)*>(eglGetProcAddress("glVertexAttribI3i"));
		VertexAttribI4i = reinterpret_cast<decltype(glVertexAttribI4i)*>(eglGetProcAddress("glVertexAttribI4i"));
		VertexAttribI1ui = reinterpret_cast<decltype(glVertexAttribI1ui)*>(eglGetProcAddress("glVertexAttribI1ui"));
		VertexAttribI2ui = reinterpret_cast<decltype(glVertexAttribI2ui)*>(eglGetProcAddress("glVertexAttribI2ui"));
		VertexAttribI3ui = reinterpret_cast<decltype(glVertexAttribI3ui)*>(eglGetProcAddress("glVertexAttribI3ui"));
		VertexAttribI4ui = reinterpret_cast<decltype(glVertexAttribI4ui)*>(eglGetProcAddress("glVertexAttribI4ui"));
		VertexAttribI1iv = reinterpret_cast<decltype(glVertexAttribI1iv)*>(eglGetProcAddress("glVertexAttribI1iv"));
		VertexAttribI2iv = reinterpret_cast<decltype(glVertexAttribI2iv)*>(eglGetProcAddress("glVertexAttribI2iv"));
		VertexAttribI3iv = reinterpret_cast<decltype(glVertexAttribI3iv)*>(eglGetProcAddress("glVertexAttribI3iv"));
		VertexAttribI4iv = reinterpret_cast<decltype(glVertexAttribI4iv)*>(eglGetProcAddress("glVertexAttribI4iv"));
		VertexAttribI1uiv = reinterpret_cast<decltype(glVertexAttribI1uiv)*>(eglGetProcAddress("glVertexAttribI1uiv"));
		VertexAttribI2uiv = reinterpret_cast<decltype(glVertexAttribI2uiv)*>(eglGetProcAddress("glVertexAttribI2uiv"));
		VertexAttribI3uiv = reinterpret_cast<decltype(glVertexAttribI3uiv)*>(eglGetProcAddress("glVertexAttribI3uiv"));
		VertexAttribI4uiv = reinterpret_cast<decltype(glVertexAttribI4uiv)*>(eglGetProcAddress("glVertexAttribI4uiv"));
		VertexAttribI4bv = reinterpret_cast<decltype(glVertexAttribI4bv)*>(eglGetProcAddress("glVertexAttribI4bv"));
		VertexAttribI4sv = reinterpret_cast<decltype(glVertexAttribI4sv)*>(eglGetProcAddress("glVertexAttribI4sv"));
		VertexAttribI4ubv = reinterpret_cast<decltype(glVertexAttribI4ubv)*>(eglGetProcAddress("glVertexAttribI4ubv"));
		VertexAttribI4usv = reinterpret_cast<decltype(glVertexAttribI4usv)*>(eglGetProcAddress("glVertexAttribI4usv"));
		GetUniformuiv = reinterpret_cast<decltype(glGetUniformuiv)*>(eglGetProcAddress("glGetUniformuiv"));
		BindFragDataLocation = reinterpret_cast<decltype(glBindFragDataLocation)*>(eglGetProcAddress("glBindFragDataLocation"));
		GetFragDataLocation = reinterpret_cast<decltype(glGetFragDataLocation)*>(eglGetProcAddress("glGetFragDataLocation"));
		Uniform1ui = reinterpret_cast<decltype(glUniform1ui)*>(eglGetProcAddress("glUniform1ui"));
		Uniform2ui = reinterpret_cast<decltype(glUniform2ui)*>(eglGetProcAddress("glUniform2ui"));
		Uniform3ui = reinterpret_cast<decltype(glUniform3ui)*>(eglGetProcAddress("glUniform3ui"));
		Uniform4ui = reinterpret_cast<decltype(glUniform4ui)*>(eglGetProcAddress("glUniform4ui"));
		Uniform1uiv = reinterpret_cast<decltype(glUniform1uiv)*>(eglGetProcAddress("glUniform1uiv"));
		Uniform2uiv = reinterpret_cast<decltype(glUniform2uiv)*>(eglGetProcAddress("glUniform2uiv"));
		Uniform3uiv = reinterpret_cast<decltype(glUniform3uiv)*>(eglGetProcAddress("glUniform3uiv"));
		Uniform4uiv = reinterpret_cast<decltype(glUniform4uiv)*>(eglGetProcAddress("glUniform4uiv"));
		TexParameterIiv = reinterpret_cast<decltype(glTexParameterIiv)*>(eglGetProcAddress("glTexParameterIiv"));
		TexParameterIuiv = reinterpret_cast<decltype(glTexParameterIuiv)*>(eglGetProcAddress("glTexParameterIuiv"));
		GetTexParameterIiv = reinterpret_cast<decltype(glGetTexParameterIiv)*>(eglGetProcAddress("glGetTexParameterIiv"));
		GetTexParameterIuiv = reinterpret_cast<decltype(glGetTexParameterIuiv)*>(eglGetProcAddress("glGetTexParameterIuiv"));
		ClearBufferiv = reinterpret_cast<decltype(glClearBufferiv)*>(eglGetProcAddress("glClearBufferiv"));
		ClearBufferuiv = reinterpret_cast<decltype(glClearBufferuiv)*>(eglGetProcAddress("glClearBufferuiv"));
		ClearBufferfv = reinterpret_cast<decltype(glClearBufferfv)*>(eglGetProcAddress("glClearBufferfv"));
		ClearBufferfi = reinterpret_cast<decltype(glClearBufferfi)*>(eglGetProcAddress("glClearBufferfi"));
		GetStringi = reinterpret_cast<decltype(glGetStringi)*>(eglGetProcAddress("glGetStringi"));
		IsRenderbuffer = reinterpret_cast<decltype(glIsRenderbuffer)*>(eglGetProcAddress("glIsRenderbuffer"));
		BindRenderbuffer = reinterpret_cast<decltype(glBindRenderbuffer)*>(eglGetProcAddress("glBindRenderbuffer"));
		DeleteRenderbuffers = reinterpret_cast<decltype(glDeleteRenderbuffers)*>(eglGetProcAddress("glDeleteRenderbuffers"));
		GenRenderbuffers = reinterpret_cast<decltype(glGenRenderbuffers)*>(eglGetProcAddress("glGenRenderbuffers"));
		RenderbufferStorage = reinterpret_cast<decltype(glRenderbufferStorage)*>(eglGetProcAddress("glRenderbufferStorage"));
		GetRenderbufferParameteriv = reinterpret_cast<decltype(glGetRenderbufferParameteriv)*>(eglGetProcAddress("glGetRenderbufferParameteriv"));
		IsFramebuffer = reinterpret_cast<decltype(glIsFramebuffer)*>(eglGetProcAddress("glIsFramebuffer"));
		BindFramebuffer = reinterpret_cast<decltype(glBindFramebuffer)*>(eglGetProcAddress("glBindFramebuffer"));
		DeleteFramebuffers = reinterpret_cast<decltype(glDeleteFramebuffers)*>(eglGetProcAddress("glDeleteFramebuffers"));
		GenFramebuffers = reinterpret_cast<decltype(glGenFramebuffers)*>(eglGetProcAddress("glGenFramebuffers"));
		CheckFramebufferStatus = reinterpret_cast<decltype(glCheckFramebufferStatus)*>(eglGetProcAddress("glCheckFramebufferStatus"));
		FramebufferTexture1D = reinterpret_cast<decltype(glFramebufferTexture1D)*>(eglGetProcAddress("glFramebufferTexture1D"));
		FramebufferTexture2D = reinterpret_cast<decltype(glFramebufferTexture2D)*>(eglGetProcAddress("glFramebufferTexture2D"));
		FramebufferTexture3D = reinterpret_cast<decltype(glFramebufferTexture3D)*>(eglGetProcAddress("glFramebufferTexture3D"));
		FramebufferRenderbuffer = reinterpret_cast<decltype(glFramebufferRenderbuffer)*>(eglGetProcAddress("glFramebufferRenderbuffer"));
		GetFramebufferAttachmentParameteriv = reinterpret_cast<decltype(glGetFramebufferAttachmentParameteriv)*>(eglGetProcAddress("glGetFramebufferAttachmentParameteriv"));
		GenerateMipmap = reinterpret_cast<decltype(glGenerateMipmap)*>(eglGetProcAddress("glGenerateMipmap"));
		BlitFramebuffer = reinterpret_cast<decltype(glBlitFramebuffer)*>(eglGetProcAddress("glBlitFramebuffer"));
		RenderbufferStorageMultisample = reinterpret_cast<decltype(glRenderbufferStorageMultisample)*>(eglGetProcAddress("glRenderbufferStorageMultisample"));
		FramebufferTextureLayer = reinterpret_cast<decltype(glFramebufferTextureLayer)*>(eglGetProcAddress("glFramebufferTextureLayer"));
		MapBufferRange = reinterpret_cast<decltype(glMapBufferRange)*>(eglGetProcAddress("glMapBufferRange"));
		FlushMappedBufferRange = reinterpret_cast<decltype(glFlushMappedBufferRange)*>(eglGetProcAddress("glFlushMappedBufferRange"));
		BindVertexArray = reinterpret_cast<decltype(glBindVertexArray)*>(eglGetProcAddress("glBindVertexArray"));
		DeleteVertexArrays = reinterpret_cast<decltype(glDeleteVertexArrays)*>(eglGetProcAddress("glDeleteVertexArrays"));
		GenVertexArrays = reinterpret_cast<decltype(glGenVertexArrays)*>(eglGetProcAddress("glGenVertexArrays"));
		IsVertexArray = reinterpret_cast<decltype(glIsVertexArray)*>(eglGetProcAddress("glIsVertexArray"));
		DrawArraysInstanced = reinterpret_cast<decltype(glDrawArraysInstanced)*>(eglGetProcAddress("glDrawArraysInstanced"));
		DrawElementsInstanced = reinterpret_cast<decltype(glDrawElementsInstanced)*>(eglGetProcAddress("glDrawElementsInstanced"));
		TexBuffer = reinterpret_cast<decltype(glTexBuffer)*>(eglGetProcAddress("glTexBuffer"));
		PrimitiveRestartIndex = reinterpret_cast<decltype(glPrimitiveRestartIndex)*>(eglGetProcAddress("glPrimitiveRestartIndex"));
		CopyBufferSubData = reinterpret_cast<decltype(glCopyBufferSubData)*>(eglGetProcAddress("glCopyBufferSubData"));
		GetUniformIndices = reinterpret_cast<decltype(glGetUniformIndices)*>(eglGetProcAddress("glGetUniformIndices"));
		GetActiveUniformsiv = reinterpret_cast<decltype(glGetActiveUniformsiv)*>(eglGetProcAddress("glGetActiveUniformsiv"));
		GetActiveUniformName = reinterpret_cast<decltype(glGetActiveUniformName)*>(eglGetProcAddress("glGetActiveUniformName"));
		GetUniformBlockIndex = reinterpret_cast<decltype(glGetUniformBlockIndex)*>(eglGetProcAddress("glGetUniformBlockIndex"));
		GetActiveUniformBlockiv = reinterpret_cast<decltype(glGetActiveUniformBlockiv)*>(eglGetProcAddress("glGetActiveUniformBlockiv"));
		GetActiveUniformBlockName = reinterpret_cast<decltype(glGetActiveUniformBlockName)*>(eglGetProcAddress("glGetActiveUniformBlockName"));
		UniformBlockBinding = reinterpret_cast<decltype(glUniformBlockBinding)*>(eglGetProcAddress("glUniformBlockBinding"));
		DrawElementsBaseVertex = reinterpret_cast<decltype(glDrawElementsBaseVertex)*>(eglGetProcAddress("glDrawElementsBaseVertex"));
		DrawRangeElementsBaseVertex = reinterpret_cast<decltype(glDrawRangeElementsBaseVertex)*>(eglGetProcAddress("glDrawRangeElementsBaseVertex"));
		DrawElementsInstancedBaseVertex = reinterpret_cast<decltype(glDrawElementsInstancedBaseVertex)*>(eglGetProcAddress("glDrawElementsInstancedBaseVertex"));
		MultiDrawElementsBaseVertex = reinterpret_cast<decltype(glMultiDrawElementsBaseVertex)*>(eglGetProcAddress("glMultiDrawElementsBaseVertex"));
		ProvokingVertex = reinterpret_cast<decltype(glProvokingVertex)*>(eglGetProcAddress("glProvokingVertex"));
		FenceSync = reinterpret_cast<decltype(glFenceSync)*>(eglGetProcAddress("glFenceSync"));
		IsSync = reinterpret_cast<decltype(glIsSync)*>(eglGetProcAddress("glIsSync"));
		DeleteSync = reinterpret_cast<decltype(glDeleteSync)*>(eglGetProcAddress("glDeleteSync"));
		ClientWaitSync = reinterpret_cast<decltype(glClientWaitSync)*>(eglGetProcAddress("glClientWaitSync"));
		WaitSync = reinterpret_cast<decltype(glWaitSync)*>(eglGetProcAddress("glWaitSync"));
		GetInteger64v = reinterpret_cast<decltype(glGetInteger64v)*>(eglGetProcAddress("glGetInteger64v"));
		GetSynciv = reinterpret_cast<decltype(glGetSynciv)*>(eglGetProcAddress("glGetSynciv"));
		GetInteger64i_v = reinterpret_cast<decltype(glGetInteger64i_v)*>(eglGetProcAddress("glGetInteger64i_v"));
		GetBufferParameteri64v = reinterpret_cast<decltype(glGetBufferParameteri64v)*>(eglGetProcAddress("glGetBufferParameteri64v"));
		FramebufferTexture = reinterpret_cast<decltype(glFramebufferTexture)*>(eglGetProcAddress("glFramebufferTexture"));
		TexImage2DMultisample = reinterpret_cast<decltype(glTexImage2DMultisample)*>(eglGetProcAddress("glTexImage2DMultisample"));
		TexImage3DMultisample = reinterpret_cast<decltype(glTexImage3DMultisample)*>(eglGetProcAddress("glTexImage3DMultisample"));
		GetMultisamplefv = reinterpret_cast<decltype(glGetMultisamplefv)*>(eglGetProcAddress("glGetMultisamplefv"));
		SampleMaski = reinterpret_cast<decltype(glSampleMaski)*>(eglGetProcAddress("glSampleMaski"));
		BindFragDataLocationIndexed = reinterpret_cast<decltype(glBindFragDataLocationIndexed)*>(eglGetProcAddress("glBindFragDataLocationIndexed"));
		GetFragDataIndex = reinterpret_cast<decltype(glGetFragDataIndex)*>(eglGetProcAddress("glGetFragDataIndex"));
		GenSamplers = reinterpret_cast<decltype(glGenSamplers)*>(eglGetProcAddress("glGenSamplers"));
		DeleteSamplers = reinterpret_cast<decltype(glDeleteSamplers)*>(eglGetProcAddress("glDeleteSamplers"));
		IsSampler = reinterpret_cast<decltype(glIsSampler)*>(eglGetProcAddress("glIsSampler"));
		BindSampler = reinterpret_cast<decltype(glBindSampler)*>(eglGetProcAddress("glBindSampler"));
		SamplerParameteri = reinterpret_cast<decltype(glSamplerParameteri)*>(eglGetProcAddress("glSamplerParameteri"));
		SamplerParameteriv = reinterpret_cast<decltype(glSamplerParameteriv)*>(eglGetProcAddress("glSamplerParameteriv"));
		SamplerParameterf = reinterpret_cast<decltype(glSamplerParameterf)*>(eglGetProcAddress("glSamplerParameterf"));
		SamplerParameterfv = reinterpret_cast<decltype(glSamplerParameterfv)*>(eglGetProcAddress("glSamplerParameterfv"));
		SamplerParameterIiv = reinterpret_cast<decltype(glSamplerParameterIiv)*>(eglGetProcAddress("glSamplerParameterIiv"));
		SamplerParameterIuiv = reinterpret_cast<decltype(glSamplerParameterIuiv)*>(eglGetProcAddress("glSamplerParameterIuiv"));
		GetSamplerParameteriv = reinterpret_cast<decltype(glGetSamplerParameteriv)*>(eglGetProcAddress("glGetSamplerParameteriv"));
		GetSamplerParameterIiv = reinterpret_cast<decltype(glGetSamplerParameterIiv)*>(eglGetProcAddress("glGetSamplerParameterIiv"));
		GetSamplerParameterfv = reinterpret_cast<decltype(glGetSamplerParameterfv)*>(eglGetProcAddress("glGetSamplerParameterfv"));
		GetSamplerParameterIuiv = reinterpret_cast<decltype(glGetSamplerParameterIuiv)*>(eglGetProcAddress("glGetSamplerParameterIuiv"));
		QueryCounter = reinterpret_cast<decltype(glQueryCounter)*>(eglGetProcAddress("glQueryCounter"));
		GetQueryObjecti64v = reinterpret_cast<decltype(glGetQueryObjecti64v)*>(eglGetProcAddress("glGetQueryObjecti64v"));
		GetQueryObjectui64v = reinterpret_cast<decltype(glGetQueryObjectui64v)*>(eglGetProcAddress("glGetQueryObjectui64v"));
		VertexAttribDivisor = reinterpret_cast<decltype(glVertexAttribDivisor)*>(eglGetProcAddress("glVertexAttribDivisor"));
		VertexAttribP1ui = reinterpret_cast<decltype(glVertexAttribP1ui)*>(eglGetProcAddress("glVertexAttribP1ui"));
		VertexAttribP1uiv = reinterpret_cast<decltype(glVertexAttribP1uiv)*>(eglGetProcAddress("glVertexAttribP1uiv"));
		VertexAttribP2ui = reinterpret_cast<decltype(glVertexAttribP2ui)*>(eglGetProcAddress("glVertexAttribP2ui"));
		VertexAttribP2uiv = reinterpret_cast<decltype(glVertexAttribP2uiv)*>(eglGetProcAddress("glVertexAttribP2uiv"));
		VertexAttribP3ui = reinterpret_cast<decltype(glVertexAttribP3ui)*>(eglGetProcAddress("glVertexAttribP3ui"));
		VertexAttribP3uiv = reinterpret_cast<decltype(glVertexAttribP3uiv)*>(eglGetProcAddress("glVertexAttribP3uiv"));
		VertexAttribP4ui = reinterpret_cast<decltype(glVertexAttribP4ui)*>(eglGetProcAddress("glVertexAttribP4ui"));
		VertexAttribP4uiv = reinterpret_cast<decltype(glVertexAttribP4uiv)*>(eglGetProcAddress("glVertexAttribP4uiv"));
		MinSampleShading = reinterpret_cast<decltype(glMinSampleShading)*>(eglGetProcAddress("glMinSampleShading"));
		BlendEquationi = reinterpret_cast<decltype(glBlendEquationi)*>(eglGetProcAddress("glBlendEquationi"));
		BlendEquationSeparatei = reinterpret_cast<decltype(glBlendEquationSeparatei)*>(eglGetProcAddress("glBlendEquationSeparatei"));
		BlendFunci = reinterpret_cast<decltype(glBlendFunci)*>(eglGetProcAddress("glBlendFunci"));
		BlendFuncSeparatei = reinterpret_cast<decltype(glBlendFuncSeparatei)*>(eglGetProcAddress("glBlendFuncSeparatei"));
		DrawArraysIndirect = reinterpret_cast<decltype(glDrawArraysIndirect)*>(eglGetProcAddress("glDrawArraysIndirect"));
		DrawElementsIndirect = reinterpret_cast<decltype(glDrawElementsIndirect)*>(eglGetProcAddress("glDrawElementsIndirect"));
		Uniform1d = reinterpret_cast<decltype(glUniform1d)*>(eglGetProcAddress("glUniform1d"));
		Uniform2d = reinterpret_cast<decltype(glUniform2d)*>(eglGetProcAddress("glUniform2d"));
		Uniform3d = reinterpret_cast<decltype(glUniform3d)*>(eglGetProcAddress("glUniform3d"));
		Uniform4d = reinterpret_cast<decltype(glUniform4d)*>(eglGetProcAddress("glUniform4d"));
		Uniform1dv = reinterpret_cast<decltype(glUniform1dv)*>(eglGetProcAddress("glUniform1dv"));
		Uniform2dv = reinterpret_cast<decltype(glUniform2dv)*>(eglGetProcAddress("glUniform2dv"));
		Uniform3dv = reinterpret_cast<decltype(glUniform3dv)*>(eglGetProcAddress("glUniform3dv"));
		Uniform4dv = reinterpret_cast<decltype(glUniform4dv)*>(eglGetProcAddress("glUniform4dv"));
		UniformMatrix2dv = reinterpret_cast<decltype(glUniformMatrix2dv)*>(eglGetProcAddress("glUniformMatrix2dv"));
		UniformMatrix3dv = reinterpret_cast<decltype(glUniformMatrix3dv)*>(eglGetProcAddress("glUniformMatrix3dv"));
		UniformMatrix4dv = reinterpret_cast<decltype(glUniformMatrix4dv)*>(eglGetProcAddress("glUniformMatrix4dv"));
		UniformMatrix2x3dv = reinterpret_cast<decltype(glUniformMatrix2x3dv)*>(eglGetProcAddress("glUniformMatrix2x3dv"));
		UniformMatrix2x4dv = reinterpret_cast<decltype(glUniformMatrix2x4dv)*>(eglGetProcAddress("glUniformMatrix2x4dv"));
		UniformMatrix3x2dv = reinterpret_cast<decltype(glUniformMatrix3x2dv)*>(eglGetProcAddress("glUniformMatrix3x2dv"));
		UniformMatrix3x4dv = reinterpret_cast<decltype(glUniformMatrix3x4dv)*>(eglGetProcAddress("glUniformMatrix3x4dv"));
		UniformMatrix4x2dv = reinterpret_cast<decltype(glUniformMatrix4x2dv)*>(eglGetProcAddress("glUniformMatrix4x2dv"));
		UniformMatrix4x3dv = reinterpret_cast<decltype(glUniformMatrix4x3dv)*>(eglGetProcAddress("glUniformMatrix4x3dv"));
		GetUniformdv = reinterpret_cast<decltype(glGetUniformdv)*>(eglGetProcAddress("glGetUniformdv"));
		GetSubroutineUniformLocation = reinterpret_cast<decltype(glGetSubroutineUniformLocation)*>(eglGetProcAddress("glGetSubroutineUniformLocation"));
		GetSubroutineIndex = reinterpret_cast<decltype(glGetSubroutineIndex)*>(eglGetProcAddress("glGetSubroutineIndex"));
		GetActiveSubroutineUniformiv = reinterpret_cast<decltype(glGetActiveSubroutineUniformiv)*>(eglGetProcAddress("glGetActiveSubroutineUniformiv"));
		GetActiveSubroutineUniformName = reinterpret_cast<decltype(glGetActiveSubroutineUniformName)*>(eglGetProcAddress("glGetActiveSubroutineUniformName"));
		GetActiveSubroutineName = reinterpret_cast<decltype(glGetActiveSubroutineName)*>(eglGetProcAddress("glGetActiveSubroutineName"));
		UniformSubroutinesuiv = reinterpret_cast<decltype(glUniformSubroutinesuiv)*>(eglGetProcAddress("glUniformSubroutinesuiv"));
		GetUniformSubroutineuiv = reinterpret_cast<decltype(glGetUniformSubroutineuiv)*>(eglGetProcAddress("glGetUniformSubroutineuiv"));
		GetProgramStageiv = reinterpret_cast<decltype(glGetProgramStageiv)*>(eglGetProcAddress("glGetProgramStageiv"));
		PatchParameteri = reinterpret_cast<decltype(glPatchParameteri)*>(eglGetProcAddress("glPatchParameteri"));
		PatchParameterfv = reinterpret_cast<decltype(glPatchParameterfv)*>(eglGetProcAddress("glPatchParameterfv"));
		BindTransformFeedback = reinterpret_cast<decltype(glBindTransformFeedback)*>(eglGetProcAddress("glBindTransformFeedback"));
		DeleteTransformFeedbacks = reinterpret_cast<decltype(glDeleteTransformFeedbacks)*>(eglGetProcAddress("glDeleteTransformFeedbacks"));
		GenTransformFeedbacks = reinterpret_cast<decltype(glGenTransformFeedbacks)*>(eglGetProcAddress("glGenTransformFeedbacks"));
		IsTransformFeedback = reinterpret_cast<decltype(glIsTransformFeedback)*>(eglGetProcAddress("glIsTransformFeedback"));
		PauseTransformFeedback = reinterpret_cast<decltype(glPauseTransformFeedback)*>(eglGetProcAddress("glPauseTransformFeedback"));
		ResumeTransformFeedback = reinterpret_cast<decltype(glResumeTransformFeedback)*>(eglGetProcAddress("glResumeTransformFeedback"));
		DrawTransformFeedback = reinterpret_cast<decltype(glDrawTransformFeedback)*>(eglGetProcAddress("glDrawTransformFeedback"));
		DrawTransformFeedbackStream = reinterpret_cast<decltype(glDrawTransformFeedbackStream)*>(eglGetProcAddress("glDrawTransformFeedbackStream"));
		BeginQueryIndexed = reinterpret_cast<decltype(glBeginQueryIndexed)*>(eglGetProcAddress("glBeginQueryIndexed"));
		EndQueryIndexed = reinterpret_cast<decltype(glEndQueryIndexed)*>(eglGetProcAddress("glEndQueryIndexed"));
		GetQueryIndexediv = reinterpret_cast<decltype(glGetQueryIndexediv)*>(eglGetProcAddress("glGetQueryIndexediv"));
		ReleaseShaderCompiler = reinterpret_cast<decltype(glReleaseShaderCompiler)*>(eglGetProcAddress("glReleaseShaderCompiler"));
		ShaderBinary = reinterpret_cast<decltype(glShaderBinary)*>(eglGetProcAddress("glShaderBinary"));
		GetShaderPrecisionFormat = reinterpret_cast<decltype(glGetShaderPrecisionFormat)*>(eglGetProcAddress("glGetShaderPrecisionFormat"));
		DepthRangef = reinterpret_cast<decltype(glDepthRangef)*>(eglGetProcAddress("glDepthRangef"));
		ClearDepthf = reinterpret_cast<decltype(glClearDepthf)*>(eglGetProcAddress("glClearDepthf"));
		GetProgramBinary = reinterpret_cast<decltype(glGetProgramBinary)*>(eglGetProcAddress("glGetProgramBinary"));
		ProgramBinary = reinterpret_cast<decltype(glProgramBinary)*>(eglGetProcAddress("glProgramBinary"));
		ProgramParameteri = reinterpret_cast<decltype(glProgramParameteri)*>(eglGetProcAddress("glProgramParameteri"));
		UseProgramStages = reinterpret_cast<decltype(glUseProgramStages)*>(eglGetProcAddress("glUseProgramStages"));
		ActiveShaderProgram = reinterpret_cast<decltype(glActiveShaderProgram)*>(eglGetProcAddress("glActiveShaderProgram"));
		CreateShaderProgramv = reinterpret_cast<decltype(glCreateShaderProgramv)*>(eglGetProcAddress("glCreateShaderProgramv"));
		BindProgramPipeline = reinterpret_cast<decltype(glBindProgramPipeline)*>(eglGetProcAddress("glBindProgramPipeline"));
		DeleteProgramPipelines = reinterpret_cast<decltype(glDeleteProgramPipelines)*>(eglGetProcAddress("glDeleteProgramPipelines"));
		GenProgramPipelines = reinterpret_cast<decltype(glGenProgramPipelines)*>(eglGetProcAddress("glGenProgramPipelines"));
		IsProgramPipeline = reinterpret_cast<decltype(glIsProgramPipeline)*>(eglGetProcAddress("glIsProgramPipeline"));
		GetProgramPipelineiv = reinterpret_cast<decltype(glGetProgramPipelineiv)*>(eglGetProcAddress("glGetProgramPipelineiv"));
		ProgramUniform1i = reinterpret_cast<decltype(glProgramUniform1i)*>(eglGetProcAddress("glProgramUniform1i"));
		ProgramUniform1iv = reinterpret_cast<decltype(glProgramUniform1iv)*>(eglGetProcAddress("glProgramUniform1iv"));
		ProgramUniform1f = reinterpret_cast<decltype(glProgramUniform1f)*>(eglGetProcAddress("glProgramUniform1f"));
		ProgramUniform1fv = reinterpret_cast<decltype(glProgramUniform1fv)*>(eglGetProcAddress("glProgramUniform1fv"));
		ProgramUniform1d = reinterpret_cast<decltype(glProgramUniform1d)*>(eglGetProcAddress("glProgramUniform1d"));
		ProgramUniform1dv = reinterpret_cast<decltype(glProgramUniform1dv)*>(eglGetProcAddress("glProgramUniform1dv"));
		ProgramUniform1ui = reinterpret_cast<decltype(glProgramUniform1ui)*>(eglGetProcAddress("glProgramUniform1ui"));
		ProgramUniform1uiv = reinterpret_cast<decltype(glProgramUniform1uiv)*>(eglGetProcAddress("glProgramUniform1uiv"));
		ProgramUniform2i = reinterpret_cast<decltype(glProgramUniform2i)*>(eglGetProcAddress("glProgramUniform2i"));
		ProgramUniform2iv = reinterpret_cast<decltype(glProgramUniform2iv)*>(eglGetProcAddress("glProgramUniform2iv"));
		ProgramUniform2f = reinterpret_cast<decltype(glProgramUniform2f)*>(eglGetProcAddress("glProgramUniform2f"));
		ProgramUniform2fv = reinterpret_cast<decltype(glProgramUniform2fv)*>(eglGetProcAddress("glProgramUniform2fv"));
		ProgramUniform2d = reinterpret_cast<decltype(glProgramUniform2d)*>(eglGetProcAddress("glProgramUniform2d"));
		ProgramUniform2dv = reinterpret_cast<decltype(glProgramUniform2dv)*>(eglGetProcAddress("glProgramUniform2dv"));
		ProgramUniform2ui = reinterpret_cast<decltype(glProgramUniform2ui)*>(eglGetProcAddress("glProgramUniform2ui"));
		ProgramUniform2uiv = reinterpret_cast<decltype(glProgramUniform2uiv)*>(eglGetProcAddress("glProgramUniform2uiv"));
		ProgramUniform3i = reinterpret_cast<decltype(glProgramUniform3i)*>(eglGetProcAddress("glProgramUniform3i"));
		ProgramUniform3iv = reinterpret_cast<decltype(glProgramUniform3iv)*>(eglGetProcAddress("glProgramUniform3iv"));
		ProgramUniform3f = reinterpret_cast<decltype(glProgramUniform3f)*>(eglGetProcAddress("glProgramUniform3f"));
		ProgramUniform3fv = reinterpret_cast<decltype(glProgramUniform3fv)*>(eglGetProcAddress("glProgramUniform3fv"));
		ProgramUniform3d = reinterpret_cast<decltype(glProgramUniform3d)*>(eglGetProcAddress("glProgramUniform3d"));
		ProgramUniform3dv = reinterpret_cast<decltype(glProgramUniform3dv)*>(eglGetProcAddress("glProgramUniform3dv"));
		ProgramUniform3ui = reinterpret_cast<decltype(glProgramUniform3ui)*>(eglGetProcAddress("glProgramUniform3ui"));
		ProgramUniform3uiv = reinterpret_cast<decltype(glProgramUniform3uiv)*>(eglGetProcAddress("glProgramUniform3uiv"));
		ProgramUniform4i = reinterpret_cast<decltype(glProgramUniform4i)*>(eglGetProcAddress("glProgramUniform4i"));
		ProgramUniform4iv = reinterpret_cast<decltype(glProgramUniform4iv)*>(eglGetProcAddress("glProgramUniform4iv"));
		ProgramUniform4f = reinterpret_cast<decltype(glProgramUniform4f)*>(eglGetProcAddress("glProgramUniform4f"));
		ProgramUniform4fv = reinterpret_cast<decltype(glProgramUniform4fv)*>(eglGetProcAddress("glProgramUniform4fv"));
		ProgramUniform4d = reinterpret_cast<decltype(glProgramUniform4d)*>(eglGetProcAddress("glProgramUniform4d"));
		ProgramUniform4dv = reinterpret_cast<decltype(glProgramUniform4dv)*>(eglGetProcAddress("glProgramUniform4dv"));
		ProgramUniform4ui = reinterpret_cast<decltype(glProgramUniform4ui)*>(eglGetProcAddress("glProgramUniform4ui"));
		ProgramUniform4uiv = reinterpret_cast<decltype(glProgramUniform4uiv)*>(eglGetProcAddress("glProgramUniform4uiv"));
		ProgramUniformMatrix2fv = reinterpret_cast<decltype(glProgramUniformMatrix2fv)*>(eglGetProcAddress("glProgramUniformMatrix2fv"));
		ProgramUniformMatrix3fv = reinterpret_cast<decltype(glProgramUniformMatrix3fv)*>(eglGetProcAddress("glProgramUniformMatrix3fv"));
		ProgramUniformMatrix4fv = reinterpret_cast<decltype(glProgramUniformMatrix4fv)*>(eglGetProcAddress("glProgramUniformMatrix4fv"));
		ProgramUniformMatrix2dv = reinterpret_cast<decltype(glProgramUniformMatrix2dv)*>(eglGetProcAddress("glProgramUniformMatrix2dv"));
		ProgramUniformMatrix3dv = reinterpret_cast<decltype(glProgramUniformMatrix3dv)*>(eglGetProcAddress("glProgramUniformMatrix3dv"));
		ProgramUniformMatrix4dv = reinterpret_cast<decltype(glProgramUniformMatrix4dv)*>(eglGetProcAddress("glProgramUniformMatrix4dv"));
		ProgramUniformMatrix2x3fv = reinterpret_cast<decltype(glProgramUniformMatrix2x3fv)*>(eglGetProcAddress("glProgramUniformMatrix2x3fv"));
		ProgramUniformMatrix3x2fv = reinterpret_cast<decltype(glProgramUniformMatrix3x2fv)*>(eglGetProcAddress("glProgramUniformMatrix3x2fv"));
		ProgramUniformMatrix2x4fv = reinterpret_cast<decltype(glProgramUniformMatrix2x4fv)*>(eglGetProcAddress("glProgramUniformMatrix2x4fv"));
		ProgramUniformMatrix4x2fv = reinterpret_cast<decltype(glProgramUniformMatrix4x2fv)*>(eglGetProcAddress("glProgramUniformMatrix4x2fv"));
		ProgramUniformMatrix3x4fv = reinterpret_cast<decltype(glProgramUniformMatrix3x4fv)*>(eglGetProcAddress("glProgramUniformMatrix3x4fv"));
		ProgramUniformMatrix4x3fv = reinterpret_cast<decltype(glProgramUniformMatrix4x3fv)*>(eglGetProcAddress("glProgramUniformMatrix4x3fv"));
		ProgramUniformMatrix2x3dv = reinterpret_cast<decltype(glProgramUniformMatrix2x3dv)*>(eglGetProcAddress("glProgramUniformMatrix2x3dv"));
		ProgramUniformMatrix3x2dv = reinterpret_cast<decltype(glProgramUniformMatrix3x2dv)*>(eglGetProcAddress("glProgramUniformMatrix3x2dv"));
		ProgramUniformMatrix2x4dv = reinterpret_cast<decltype(glProgramUniformMatrix2x4dv)*>(eglGetProcAddress("glProgramUniformMatrix2x4dv"));
		ProgramUniformMatrix4x2dv = reinterpret_cast<decltype(glProgramUniformMatrix4x2dv)*>(eglGetProcAddress("glProgramUniformMatrix4x2dv"));
		ProgramUniformMatrix3x4dv = reinterpret_cast<decltype(glProgramUniformMatrix3x4dv)*>(eglGetProcAddress("glProgramUniformMatrix3x4dv"));
		ProgramUniformMatrix4x3dv = reinterpret_cast<decltype(glProgramUniformMatrix4x3dv)*>(eglGetProcAddress("glProgramUniformMatrix4x3dv"));
		ValidateProgramPipeline = reinterpret_cast<decltype(glValidateProgramPipeline)*>(eglGetProcAddress("glValidateProgramPipeline"));
		GetProgramPipelineInfoLog = reinterpret_cast<decltype(glGetProgramPipelineInfoLog)*>(eglGetProcAddress("glGetProgramPipelineInfoLog"));
		VertexAttribL1d = reinterpret_cast<decltype(glVertexAttribL1d)*>(eglGetProcAddress("glVertexAttribL1d"));
		VertexAttribL2d = reinterpret_cast<decltype(glVertexAttribL2d)*>(eglGetProcAddress("glVertexAttribL2d"));
		VertexAttribL3d = reinterpret_cast<decltype(glVertexAttribL3d)*>(eglGetProcAddress("glVertexAttribL3d"));
		VertexAttribL4d = reinterpret_cast<decltype(glVertexAttribL4d)*>(eglGetProcAddress("glVertexAttribL4d"));
		VertexAttribL1dv = reinterpret_cast<decltype(glVertexAttribL1dv)*>(eglGetProcAddress("glVertexAttribL1dv"));
		VertexAttribL2dv = reinterpret_cast<decltype(glVertexAttribL2dv)*>(eglGetProcAddress("glVertexAttribL2dv"));
		VertexAttribL3dv = reinterpret_cast<decltype(glVertexAttribL3dv)*>(eglGetProcAddress("glVertexAttribL3dv"));
		VertexAttribL4dv = reinterpret_cast<decltype(glVertexAttribL4dv)*>(eglGetProcAddress("glVertexAttribL4dv"));
		VertexAttribLPointer = reinterpret_cast<decltype(glVertexAttribLPointer)*>(eglGetProcAddress("glVertexAttribLPointer"));
		GetVertexAttribLdv = reinterpret_cast<decltype(glGetVertexAttribLdv)*>(eglGetProcAddress("glGetVertexAttribLdv"));
		ViewportArrayv = reinterpret_cast<decltype(glViewportArrayv)*>(eglGetProcAddress("glViewportArrayv"));
		ViewportIndexedf = reinterpret_cast<decltype(glViewportIndexedf)*>(eglGetProcAddress("glViewportIndexedf"));
		ViewportIndexedfv = reinterpret_cast<decltype(glViewportIndexedfv)*>(eglGetProcAddress("glViewportIndexedfv"));
		ScissorArrayv = reinterpret_cast<decltype(glScissorArrayv)*>(eglGetProcAddress("glScissorArrayv"));
		ScissorIndexed = reinterpret_cast<decltype(glScissorIndexed)*>(eglGetProcAddress("glScissorIndexed"));
		ScissorIndexedv = reinterpret_cast<decltype(glScissorIndexedv)*>(eglGetProcAddress("glScissorIndexedv"));
		DepthRangeArrayv = reinterpret_cast<decltype(glDepthRangeArrayv)*>(eglGetProcAddress("glDepthRangeArrayv"));
		DepthRangeIndexed = reinterpret_cast<decltype(glDepthRangeIndexed)*>(eglGetProcAddress("glDepthRangeIndexed"));
		GetFloati_v = reinterpret_cast<decltype(glGetFloati_v)*>(eglGetProcAddress("glGetFloati_v"));
		GetDoublei_v = reinterpret_cast<decltype(glGetDoublei_v)*>(eglGetProcAddress("glGetDoublei_v"));
		DrawArraysInstancedBaseInstance = reinterpret_cast<decltype(glDrawArraysInstancedBaseInstance)*>(eglGetProcAddress("glDrawArraysInstancedBaseInstance"));
		DrawElementsInstancedBaseInstance = reinterpret_cast<decltype(glDrawElementsInstancedBaseInstance)*>(eglGetProcAddress("glDrawElementsInstancedBaseInstance"));
		DrawElementsInstancedBaseVertexBaseInstance = reinterpret_cast<decltype(glDrawElementsInstancedBaseVertexBaseInstance)*>(eglGetProcAddress("glDrawElementsInstancedBaseVertexBaseInstance"));
		GetInternalformativ = reinterpret_cast<decltype(glGetInternalformativ)*>(eglGetProcAddress("glGetInternalformativ"));
		GetActiveAtomicCounterBufferiv = reinterpret_cast<decltype(glGetActiveAtomicCounterBufferiv)*>(eglGetProcAddress("glGetActiveAtomicCounterBufferiv"));
		BindImageTexture = reinterpret_cast<decltype(glBindImageTexture)*>(eglGetProcAddress("glBindImageTexture"));
		MemoryBarrier = reinterpret_cast<decltype(glMemoryBarrier)*>(eglGetProcAddress("glMemoryBarrier"));
		TexStorage1D = reinterpret_cast<decltype(glTexStorage1D)*>(eglGetProcAddress("glTexStorage1D"));
		TexStorage2D = reinterpret_cast<decltype(glTexStorage2D)*>(eglGetProcAddress("glTexStorage2D"));
		TexStorage3D = reinterpret_cast<decltype(glTexStorage3D)*>(eglGetProcAddress("glTexStorage3D"));
		DrawTransformFeedbackInstanced = reinterpret_cast<decltype(glDrawTransformFeedbackInstanced)*>(eglGetProcAddress("glDrawTransformFeedbackInstanced"));
		DrawTransformFeedbackStreamInstanced = reinterpret_cast<decltype(glDrawTransformFeedbackStreamInstanced)*>(eglGetProcAddress("glDrawTransformFeedbackStreamInstanced"));
		ClearBufferData = reinterpret_cast<decltype(glClearBufferData)*>(eglGetProcAddress("glClearBufferData"));
		ClearBufferSubData = reinterpret_cast<decltype(glClearBufferSubData)*>(eglGetProcAddress("glClearBufferSubData"));
		DispatchCompute = reinterpret_cast<decltype(glDispatchCompute)*>(eglGetProcAddress("glDispatchCompute"));
		DispatchComputeIndirect = reinterpret_cast<decltype(glDispatchComputeIndirect)*>(eglGetProcAddress("glDispatchComputeIndirect"));
		CopyImageSubData = reinterpret_cast<decltype(glCopyImageSubData)*>(eglGetProcAddress("glCopyImageSubData"));
		FramebufferParameteri = reinterpret_cast<decltype(glFramebufferParameteri)*>(eglGetProcAddress("glFramebufferParameteri"));
		GetFramebufferParameteriv = reinterpret_cast<decltype(glGetFramebufferParameteriv)*>(eglGetProcAddress("glGetFramebufferParameteriv"));
		GetInternalformati64v = reinterpret_cast<decltype(glGetInternalformati64v)*>(eglGetProcAddress("glGetInternalformati64v"));
		InvalidateTexSubImage = reinterpret_cast<decltype(glInvalidateTexSubImage)*>(eglGetProcAddress("glInvalidateTexSubImage"));
		InvalidateTexImage = reinterpret_cast<decltype(glInvalidateTexImage)*>(eglGetProcAddress("glInvalidateTexImage"));
		InvalidateBufferSubData = reinterpret_cast<decltype(glInvalidateBufferSubData)*>(eglGetProcAddress("glInvalidateBufferSubData"));
		InvalidateBufferData = reinterpret_cast<decltype(glInvalidateBufferData)*>(eglGetProcAddress("glInvalidateBufferData"));
		InvalidateFramebuffer = reinterpret_cast<decltype(glInvalidateFramebuffer)*>(eglGetProcAddress("glInvalidateFramebuffer"));
		InvalidateSubFramebuffer = reinterpret_cast<decltype(glInvalidateSubFramebuffer)*>(eglGetProcAddress("glInvalidateSubFramebuffer"));
		MultiDrawArraysIndirect = reinterpret_cast<decltype(glMultiDrawArraysIndirect)*>(eglGetProcAddress("glMultiDrawArraysIndirect"));
		MultiDrawElementsIndirect = reinterpret_cast<decltype(glMultiDrawElementsIndirect)*>(eglGetProcAddress("glMultiDrawElementsIndirect"));
		GetProgramInterfaceiv = reinterpret_cast<decltype(glGetProgramInterfaceiv)*>(eglGetProcAddress("glGetProgramInterfaceiv"));
		GetProgramResourceIndex = reinterpret_cast<decltype(glGetProgramResourceIndex)*>(eglGetProcAddress("glGetProgramResourceIndex"));
		GetProgramResourceName = reinterpret_cast<decltype(glGetProgramResourceName)*>(eglGetProcAddress("glGetProgramResourceName"));
		GetProgramResourceiv = reinterpret_cast<decltype(glGetProgramResourceiv)*>(eglGetProcAddress("glGetProgramResourceiv"));
		GetProgramResourceLocation = reinterpret_cast<decltype(glGetProgramResourceLocation)*>(eglGetProcAddress("glGetProgramResourceLocation"));
		GetProgramResourceLocationIndex = reinterpret_cast<decltype(glGetProgramResourceLocationIndex)*>(eglGetProcAddress("glGetProgramResourceLocationIndex"));
		ShaderStorageBlockBinding = reinterpret_cast<decltype(glShaderStorageBlockBinding)*>(eglGetProcAddress("glShaderStorageBlockBinding"));
		TexBufferRange = reinterpret_cast<decltype(glTexBufferRange)*>(eglGetProcAddress("glTexBufferRange"));
		TexStorage2DMultisample = reinterpret_cast<decltype(glTexStorage2DMultisample)*>(eglGetProcAddress("glTexStorage2DMultisample"));
		TexStorage3DMultisample = reinterpret_cast<decltype(glTexStorage3DMultisample)*>(eglGetProcAddress("glTexStorage3DMultisample"));
		TextureView = reinterpret_cast<decltype(glTextureView)*>(eglGetProcAddress("glTextureView"));
		BindVertexBuffer = reinterpret_cast<decltype(glBindVertexBuffer)*>(eglGetProcAddress("glBindVertexBuffer"));
		VertexAttribFormat = reinterpret_cast<decltype(glVertexAttribFormat)*>(eglGetProcAddress("glVertexAttribFormat"));
		VertexAttribIFormat = reinterpret_cast<decltype(glVertexAttribIFormat)*>(eglGetProcAddress("glVertexAttribIFormat"));
		VertexAttribLFormat = reinterpret_cast<decltype(glVertexAttribLFormat)*>(eglGetProcAddress("glVertexAttribLFormat"));
		VertexAttribBinding = reinterpret_cast<decltype(glVertexAttribBinding)*>(eglGetProcAddress("glVertexAttribBinding"));
		VertexBindingDivisor = reinterpret_cast<decltype(glVertexBindingDivisor)*>(eglGetProcAddress("glVertexBindingDivisor"));
		DebugMessageControl = reinterpret_cast<decltype(glDebugMessageControl)*>(eglGetProcAddress("glDebugMessageControl"));
		DebugMessageInsert = reinterpret_cast<decltype(glDebugMessageInsert)*>(eglGetProcAddress("glDebugMessageInsert"));
		DebugMessageCallback = reinterpret_cast<decltype(glDebugMessageCallback)*>(eglGetProcAddress("glDebugMessageCallback"));
		GetDebugMessageLog = reinterpret_cast<decltype(glGetDebugMessageLog)*>(eglGetProcAddress("glGetDebugMessageLog"));
		PushDebugGroup = reinterpret_cast<decltype(glPushDebugGroup)*>(eglGetProcAddress("glPushDebugGroup"));
		PopDebugGroup = reinterpret_cast<decltype(glPopDebugGroup)*>(eglGetProcAddress("glPopDebugGroup"));
		ObjectLabel = reinterpret_cast<decltype(glObjectLabel)*>(eglGetProcAddress("glObjectLabel"));
		GetObjectLabel = reinterpret_cast<decltype(glGetObjectLabel)*>(eglGetProcAddress("glGetObjectLabel"));
		ObjectPtrLabel = reinterpret_cast<decltype(glObjectPtrLabel)*>(eglGetProcAddress("glObjectPtrLabel"));
		GetObjectPtrLabel = reinterpret_cast<decltype(glGetObjectPtrLabel)*>(eglGetProcAddress("glGetObjectPtrLabel"));
		BufferStorage = reinterpret_cast<decltype(glBufferStorage)*>(eglGetProcAddress("glBufferStorage"));
		ClearTexImage = reinterpret_cast<decltype(glClearTexImage)*>(eglGetProcAddress("glClearTexImage"));
		ClearTexSubImage = reinterpret_cast<decltype(glClearTexSubImage)*>(eglGetProcAddress("glClearTexSubImage"));
		BindBuffersBase = reinterpret_cast<decltype(glBindBuffersBase)*>(eglGetProcAddress("glBindBuffersBase"));
		BindBuffersRange = reinterpret_cast<decltype(glBindBuffersRange)*>(eglGetProcAddress("glBindBuffersRange"));
		BindTextures = reinterpret_cast<decltype(glBindTextures)*>(eglGetProcAddress("glBindTextures"));
		BindSamplers = reinterpret_cast<decltype(glBindSamplers)*>(eglGetProcAddress("glBindSamplers"));
		BindImageTextures = reinterpret_cast<decltype(glBindImageTextures)*>(eglGetProcAddress("glBindImageTextures"));
		BindVertexBuffers = reinterpret_cast<decltype(glBindVertexBuffers)*>(eglGetProcAddress("glBindVertexBuffers"));
		ClipControl = reinterpret_cast<decltype(glClipControl)*>(eglGetProcAddress("glClipControl"));
		CreateTransformFeedbacks = reinterpret_cast<decltype(glCreateTransformFeedbacks)*>(eglGetProcAddress("glCreateTransformFeedbacks"));
		TransformFeedbackBufferBase = reinterpret_cast<decltype(glTransformFeedbackBufferBase)*>(eglGetProcAddress("glTransformFeedbackBufferBase"));
		TransformFeedbackBufferRange = reinterpret_cast<decltype(glTransformFeedbackBufferRange)*>(eglGetProcAddress("glTransformFeedbackBufferRange"));
		GetTransformFeedbackiv = reinterpret_cast<decltype(glGetTransformFeedbackiv)*>(eglGetProcAddress("glGetTransformFeedbackiv"));
		GetTransformFeedbacki_v = reinterpret_cast<decltype(glGetTransformFeedbacki_v)*>(eglGetProcAddress("glGetTransformFeedbacki_v"));
		GetTransformFeedbacki64_v = reinterpret_cast<decltype(glGetTransformFeedbacki64_v)*>(eglGetProcAddress("glGetTransformFeedbacki64_v"));
		CreateBuffers = reinterpret_cast<decltype(glCreateBuffers)*>(eglGetProcAddress("glCreateBuffers"));
		NamedBufferStorage = reinterpret_cast<decltype(glNamedBufferStorage)*>(eglGetProcAddress("glNamedBufferStorage"));
		NamedBufferData = reinterpret_cast<decltype(glNamedBufferData)*>(eglGetProcAddress("glNamedBufferData"));
		NamedBufferSubData = reinterpret_cast<decltype(glNamedBufferSubData)*>(eglGetProcAddress("glNamedBufferSubData"));
		CopyNamedBufferSubData = reinterpret_cast<decltype(glCopyNamedBufferSubData)*>(eglGetProcAddress("glCopyNamedBufferSubData"));
		ClearNamedBufferData = reinterpret_cast<decltype(glClearNamedBufferData)*>(eglGetProcAddress("glClearNamedBufferData"));
		ClearNamedBufferSubData = reinterpret_cast<decltype(glClearNamedBufferSubData)*>(eglGetProcAddress("glClearNamedBufferSubData"));
		MapNamedBuffer = reinterpret_cast<decltype(glMapNamedBuffer)*>(eglGetProcAddress("glMapNamedBuffer"));
		MapNamedBufferRange = reinterpret_cast<decltype(glMapNamedBufferRange)*>(eglGetProcAddress("glMapNamedBufferRange"));
		UnmapNamedBuffer = reinterpret_cast<decltype(glUnmapNamedBuffer)*>(eglGetProcAddress("glUnmapNamedBuffer"));
		FlushMappedNamedBufferRange = reinterpret_cast<decltype(glFlushMappedNamedBufferRange)*>(eglGetProcAddress("glFlushMappedNamedBufferRange"));
		GetNamedBufferParameteriv = reinterpret_cast<decltype(glGetNamedBufferParameteriv)*>(eglGetProcAddress("glGetNamedBufferParameteriv"));
		GetNamedBufferParameteri64v = reinterpret_cast<decltype(glGetNamedBufferParameteri64v)*>(eglGetProcAddress("glGetNamedBufferParameteri64v"));
		GetNamedBufferPointerv = reinterpret_cast<decltype(glGetNamedBufferPointerv)*>(eglGetProcAddress("glGetNamedBufferPointerv"));
		GetNamedBufferSubData = reinterpret_cast<decltype(glGetNamedBufferSubData)*>(eglGetProcAddress("glGetNamedBufferSubData"));
		CreateFramebuffers = reinterpret_cast<decltype(glCreateFramebuffers)*>(eglGetProcAddress("glCreateFramebuffers"));
		NamedFramebufferRenderbuffer = reinterpret_cast<decltype(glNamedFramebufferRenderbuffer)*>(eglGetProcAddress("glNamedFramebufferRenderbuffer"));
		NamedFramebufferParameteri = reinterpret_cast<decltype(glNamedFramebufferParameteri)*>(eglGetProcAddress("glNamedFramebufferParameteri"));
		NamedFramebufferTexture = reinterpret_cast<decltype(glNamedFramebufferTexture)*>(eglGetProcAddress("glNamedFramebufferTexture"));
		NamedFramebufferTextureLayer = reinterpret_cast<decltype(glNamedFramebufferTextureLayer)*>(eglGetProcAddress("glNamedFramebufferTextureLayer"));
		NamedFramebufferDrawBuffer = reinterpret_cast<decltype(glNamedFramebufferDrawBuffer)*>(eglGetProcAddress("glNamedFramebufferDrawBuffer"));
		NamedFramebufferDrawBuffers = reinterpret_cast<decltype(glNamedFramebufferDrawBuffers)*>(eglGetProcAddress("glNamedFramebufferDrawBuffers"));
		NamedFramebufferReadBuffer = reinterpret_cast<decltype(glNamedFramebufferReadBuffer)*>(eglGetProcAddress("glNamedFramebufferReadBuffer"));
		InvalidateNamedFramebufferData = reinterpret_cast<decltype(glInvalidateNamedFramebufferData)*>(eglGetProcAddress("glInvalidateNamedFramebufferData"));
		InvalidateNamedFramebufferSubData = reinterpret_cast<decltype(glInvalidateNamedFramebufferSubData)*>(eglGetProcAddress("glInvalidateNamedFramebufferSubData"));
		ClearNamedFramebufferiv = reinterpret_cast<decltype(glClearNamedFramebufferiv)*>(eglGetProcAddress("glClearNamedFramebufferiv"));
		ClearNamedFramebufferuiv = reinterpret_cast<decltype(glClearNamedFramebufferuiv)*>(eglGetProcAddress("glClearNamedFramebufferuiv"));
		ClearNamedFramebufferfv = reinterpret_cast<decltype(glClearNamedFramebufferfv)*>(eglGetProcAddress("glClearNamedFramebufferfv"));
		ClearNamedFramebufferfi = reinterpret_cast<decltype(glClearNamedFramebufferfi)*>(eglGetProcAddress("glClearNamedFramebufferfi"));
		BlitNamedFramebuffer = reinterpret_cast<decltype(glBlitNamedFramebuffer)*>(eglGetProcAddress("glBlitNamedFramebuffer"));
		CheckNamedFramebufferStatus = reinterpret_cast<decltype(glCheckNamedFramebufferStatus)*>(eglGetProcAddress("glCheckNamedFramebufferStatus"));
		GetNamedFramebufferParameteriv = reinterpret_cast<decltype(glGetNamedFramebufferParameteriv)*>(eglGetProcAddress("glGetNamedFramebufferParameteriv"));
		GetNamedFramebufferAttachmentParameteriv = reinterpret_cast<decltype(glGetNamedFramebufferAttachmentParameteriv)*>(eglGetProcAddress("glGetNamedFramebufferAttachmentParameteriv"));
		CreateRenderbuffers = reinterpret_cast<decltype(glCreateRenderbuffers)*>(eglGetProcAddress("glCreateRenderbuffers"));
		NamedRenderbufferStorage = reinterpret_cast<decltype(glNamedRenderbufferStorage)*>(eglGetProcAddress("glNamedRenderbufferStorage"));
		NamedRenderbufferStorageMultisample = reinterpret_cast<decltype(glNamedRenderbufferStorageMultisample)*>(eglGetProcAddress("glNamedRenderbufferStorageMultisample"));
		GetNamedRenderbufferParameteriv = reinterpret_cast<decltype(glGetNamedRenderbufferParameteriv)*>(eglGetProcAddress("glGetNamedRenderbufferParameteriv"));
		CreateTextures = reinterpret_cast<decltype(glCreateTextures)*>(eglGetProcAddress("glCreateTextures"));
		TextureBuffer = reinterpret_cast<decltype(glTextureBuffer)*>(eglGetProcAddress("glTextureBuffer"));
		TextureBufferRange = reinterpret_cast<decltype(glTextureBufferRange)*>(eglGetProcAddress("glTextureBufferRange"));
		TextureStorage1D = reinterpret_cast<decltype(glTextureStorage1D)*>(eglGetProcAddress("glTextureStorage1D"));
		TextureStorage2D = reinterpret_cast<decltype(glTextureStorage2D)*>(eglGetProcAddress("glTextureStorage2D"));
		TextureStorage3D = reinterpret_cast<decltype(glTextureStorage3D)*>(eglGetProcAddress("glTextureStorage3D"));
		TextureStorage2DMultisample = reinterpret_cast<decltype(glTextureStorage2DMultisample)*>(eglGetProcAddress("glTextureStorage2DMultisample"));
		TextureStorage3DMultisample = reinterpret_cast<decltype(glTextureStorage3DMultisample)*>(eglGetProcAddress("glTextureStorage3DMultisample"));
		TextureSubImage1D = reinterpret_cast<decltype(glTextureSubImage1D)*>(eglGetProcAddress("glTextureSubImage1D"));
		TextureSubImage2D = reinterpret_cast<decltype(glTextureSubImage2D)*>(eglGetProcAddress("glTextureSubImage2D"));
		TextureSubImage3D = reinterpret_cast<decltype(glTextureSubImage3D)*>(eglGetProcAddress("glTextureSubImage3D"));
		CompressedTextureSubImage1D = reinterpret_cast<decltype(glCompressedTextureSubImage1D)*>(eglGetProcAddress("glCompressedTextureSubImage1D"));
		CompressedTextureSubImage2D = reinterpret_cast<decltype(glCompressedTextureSubImage2D)*>(eglGetProcAddress("glCompressedTextureSubImage2D"));
		CompressedTextureSubImage3D = reinterpret_cast<decltype(glCompressedTextureSubImage3D)*>(eglGetProcAddress("glCompressedTextureSubImage3D"));
		CopyTextureSubImage1D = reinterpret_cast<decltype(glCopyTextureSubImage1D)*>(eglGetProcAddress("glCopyTextureSubImage1D"));
		CopyTextureSubImage2D = reinterpret_cast<decltype(glCopyTextureSubImage2D)*>(eglGetProcAddress("glCopyTextureSubImage2D"));
		CopyTextureSubImage3D = reinterpret_cast<decltype(glCopyTextureSubImage3D)*>(eglGetProcAddress("glCopyTextureSubImage3D"));
		TextureParameterf = reinterpret_cast<decltype(glTextureParameterf)*>(eglGetProcAddress("glTextureParameterf"));
		TextureParameterfv = reinterpret_cast<decltype(glTextureParameterfv)*>(eglGetProcAddress("glTextureParameterfv"));
		TextureParameteri = reinterpret_cast<decltype(glTextureParameteri)*>(eglGetProcAddress("glTextureParameteri"));
		TextureParameterIiv = reinterpret_cast<decltype(glTextureParameterIiv)*>(eglGetProcAddress("glTextureParameterIiv"));
		TextureParameterIuiv = reinterpret_cast<decltype(glTextureParameterIuiv)*>(eglGetProcAddress("glTextureParameterIuiv"));
		TextureParameteriv = reinterpret_cast<decltype(glTextureParameteriv)*>(eglGetProcAddress("glTextureParameteriv"));
		GenerateTextureMipmap = reinterpret_cast<decltype(glGenerateTextureMipmap)*>(eglGetProcAddress("glGenerateTextureMipmap"));
		BindTextureUnit = reinterpret_cast<decltype(glBindTextureUnit)*>(eglGetProcAddress("glBindTextureUnit"));
		GetTextureImage = reinterpret_cast<decltype(glGetTextureImage)*>(eglGetProcAddress("glGetTextureImage"));
		GetCompressedTextureImage = reinterpret_cast<decltype(glGetCompressedTextureImage)*>(eglGetProcAddress("glGetCompressedTextureImage"));
		GetTextureLevelParameterfv = reinterpret_cast<decltype(glGetTextureLevelParameterfv)*>(eglGetProcAddress("glGetTextureLevelParameterfv"));
		GetTextureLevelParameteriv = reinterpret_cast<decltype(glGetTextureLevelParameteriv)*>(eglGetProcAddress("glGetTextureLevelParameteriv"));
		GetTextureParameterfv = reinterpret_cast<decltype(glGetTextureParameterfv)*>(eglGetProcAddress("glGetTextureParameterfv"));
		GetTextureParameterIiv = reinterpret_cast<decltype(glGetTextureParameterIiv)*>(eglGetProcAddress("glGetTextureParameterIiv"));
		GetTextureParameterIuiv = reinterpret_cast<decltype(glGetTextureParameterIuiv)*>(eglGetProcAddress("glGetTextureParameterIuiv"));
		GetTextureParameteriv = reinterpret_cast<decltype(glGetTextureParameteriv)*>(eglGetProcAddress("glGetTextureParameteriv"));
		CreateVertexArrays = reinterpret_cast<decltype(glCreateVertexArrays)*>(eglGetProcAddress("glCreateVertexArrays"));
		DisableVertexArrayAttrib = reinterpret_cast<decltype(glDisableVertexArrayAttrib)*>(eglGetProcAddress("glDisableVertexArrayAttrib"));
		EnableVertexArrayAttrib = reinterpret_cast<decltype(glEnableVertexArrayAttrib)*>(eglGetProcAddress("glEnableVertexArrayAttrib"));
		VertexArrayElementBuffer = reinterpret_cast<decltype(glVertexArrayElementBuffer)*>(eglGetProcAddress("glVertexArrayElementBuffer"));
		VertexArrayVertexBuffer = reinterpret_cast<decltype(glVertexArrayVertexBuffer)*>(eglGetProcAddress("glVertexArrayVertexBuffer"));
		VertexArrayVertexBuffers = reinterpret_cast<decltype(glVertexArrayVertexBuffers)*>(eglGetProcAddress("glVertexArrayVertexBuffers"));
		VertexArrayAttribBinding = reinterpret_cast<decltype(glVertexArrayAttribBinding)*>(eglGetProcAddress("glVertexArrayAttribBinding"));
		VertexArrayAttribFormat = reinterpret_cast<decltype(glVertexArrayAttribFormat)*>(eglGetProcAddress("glVertexArrayAttribFormat"));
		VertexArrayAttribIFormat = reinterpret_cast<decltype(glVertexArrayAttribIFormat)*>(eglGetProcAddress("glVertexArrayAttribIFormat"));
		VertexArrayAttribLFormat = reinterpret_cast<decltype(glVertexArrayAttribLFormat)*>(eglGetProcAddress("glVertexArrayAttribLFormat"));
		VertexArrayBindingDivisor = reinterpret_cast<decltype(glVertexArrayBindingDivisor)*>(eglGetProcAddress("glVertexArrayBindingDivisor"));
		GetVertexArrayiv = reinterpret_cast<decltype(glGetVertexArrayiv)*>(eglGetProcAddress("glGetVertexArrayiv"));
		GetVertexArrayIndexediv = reinterpret_cast<decltype(glGetVertexArrayIndexediv)*>(eglGetProcAddress("glGetVertexArrayIndexediv"));
		GetVertexArrayIndexed64iv = reinterpret_cast<decltype(glGetVertexArrayIndexed64iv)*>(eglGetProcAddress("glGetVertexArrayIndexed64iv"));
		CreateSamplers = reinterpret_cast<decltype(glCreateSamplers)*>(eglGetProcAddress("glCreateSamplers"));
		CreateProgramPipelines = reinterpret_cast<decltype(glCreateProgramPipelines)*>(eglGetProcAddress("glCreateProgramPipelines"));
		CreateQueries = reinterpret_cast<decltype(glCreateQueries)*>(eglGetProcAddress("glCreateQueries"));
		GetQueryBufferObjecti64v = reinterpret_cast<decltype(glGetQueryBufferObjecti64v)*>(eglGetProcAddress("glGetQueryBufferObjecti64v"));
		GetQueryBufferObjectiv = reinterpret_cast<decltype(glGetQueryBufferObjectiv)*>(eglGetProcAddress("glGetQueryBufferObjectiv"));
		GetQueryBufferObjectui64v = reinterpret_cast<decltype(glGetQueryBufferObjectui64v)*>(eglGetProcAddress("glGetQueryBufferObjectui64v"));
		GetQueryBufferObjectuiv = reinterpret_cast<decltype(glGetQueryBufferObjectuiv)*>(eglGetProcAddress("glGetQueryBufferObjectuiv"));
		MemoryBarrierByRegion = reinterpret_cast<decltype(glMemoryBarrierByRegion)*>(eglGetProcAddress("glMemoryBarrierByRegion"));
		GetTextureSubImage = reinterpret_cast<decltype(glGetTextureSubImage)*>(eglGetProcAddress("glGetTextureSubImage"));
		GetCompressedTextureSubImage = reinterpret_cast<decltype(glGetCompressedTextureSubImage)*>(eglGetProcAddress("glGetCompressedTextureSubImage"));
		GetGraphicsResetStatus = reinterpret_cast<decltype(glGetGraphicsResetStatus)*>(eglGetProcAddress("glGetGraphicsResetStatus"));
		GetnCompressedTexImage = reinterpret_cast<decltype(glGetnCompressedTexImage)*>(eglGetProcAddress("glGetnCompressedTexImage"));
		GetnTexImage = reinterpret_cast<decltype(glGetnTexImage)*>(eglGetProcAddress("glGetnTexImage"));
		GetnUniformdv = reinterpret_cast<decltype(glGetnUniformdv)*>(eglGetProcAddress("glGetnUniformdv"));
		GetnUniformfv = reinterpret_cast<decltype(glGetnUniformfv)*>(eglGetProcAddress("glGetnUniformfv"));
		GetnUniformiv = reinterpret_cast<decltype(glGetnUniformiv)*>(eglGetProcAddress("glGetnUniformiv"));
		GetnUniformuiv = reinterpret_cast<decltype(glGetnUniformuiv)*>(eglGetProcAddress("glGetnUniformuiv"));
		ReadnPixels = reinterpret_cast<decltype(glReadnPixels)*>(eglGetProcAddress("glReadnPixels"));
		TextureBarrier = reinterpret_cast<decltype(glTextureBarrier)*>(eglGetProcAddress("glTextureBarrier"));

		if (CullFace == nullptr ||
		    FrontFace == nullptr ||
		    Hint == nullptr ||
		    LineWidth == nullptr ||
		    PointSize == nullptr ||
		    PolygonMode == nullptr ||
		    Scissor == nullptr ||
		    TexParameterf == nullptr ||
		    TexParameterfv == nullptr ||
		    TexParameteri == nullptr ||
		    TexParameteriv == nullptr ||
		    TexImage1D == nullptr ||
		    TexImage2D == nullptr ||
		    DrawBuffer == nullptr ||
		    Clear == nullptr ||
		    ClearColor == nullptr ||
		    ClearStencil == nullptr ||
		    ClearDepth == nullptr ||
		    StencilMask == nullptr ||
		    ColorMask == nullptr ||
		    DepthMask == nullptr ||
		    Disable == nullptr ||
		    Enable == nullptr ||
		    Finish == nullptr ||
		    Flush == nullptr ||
		    BlendFunc == nullptr ||
		    LogicOp == nullptr ||
		    StencilFunc == nullptr ||
		    StencilOp == nullptr ||
		    DepthFunc == nullptr ||
		    PixelStoref == nullptr ||
		    PixelStorei == nullptr ||
		    ReadBuffer == nullptr ||
		    ReadPixels == nullptr ||
		    GetBooleanv == nullptr ||
		    GetDoublev == nullptr ||
		    GetError == nullptr ||
		    GetFloatv == nullptr ||
		    GetIntegerv == nullptr ||
		    GetString == nullptr ||
		    GetTexImage == nullptr ||
		    GetTexParameterfv == nullptr ||
		    GetTexParameteriv == nullptr ||
		    GetTexLevelParameterfv == nullptr ||
		    GetTexLevelParameteriv == nullptr ||
		    IsEnabled == nullptr ||
		    DepthRange == nullptr ||
		    Viewport == nullptr ||
		    DrawArrays == nullptr ||
		    DrawElements == nullptr ||
		    PolygonOffset == nullptr ||
		    CopyTexImage1D == nullptr ||
		    CopyTexImage2D == nullptr ||
		    CopyTexSubImage1D == nullptr ||
		    CopyTexSubImage2D == nullptr ||
		    TexSubImage1D == nullptr ||
		    TexSubImage2D == nullptr ||
		    BindTexture == nullptr ||
		    DeleteTextures == nullptr ||
		    GenTextures == nullptr ||
		    IsTexture == nullptr ||
		    DrawRangeElements == nullptr ||
		    TexImage3D == nullptr ||
		    TexSubImage3D == nullptr ||
		    CopyTexSubImage3D == nullptr ||
		    ActiveTexture == nullptr ||
		    SampleCoverage == nullptr ||
		    CompressedTexImage3D == nullptr ||
		    CompressedTexImage2D == nullptr ||
		    CompressedTexImage1D == nullptr ||
		    CompressedTexSubImage3D == nullptr ||
		    CompressedTexSubImage2D == nullptr ||
		    CompressedTexSubImage1D == nullptr ||
		    GetCompressedTexImage == nullptr ||
		    BlendFuncSeparate == nullptr ||
		    MultiDrawArrays == nullptr ||
		    MultiDrawElements == nullptr ||
		    PointParameterf == nullptr ||
		    PointParameterfv == nullptr ||
		    PointParameteri == nullptr ||
		    PointParameteriv == nullptr ||
		    BlendColor == nullptr ||
		    BlendEquation == nullptr ||
		    GenQueries == nullptr ||
		    DeleteQueries == nullptr ||
		    IsQuery == nullptr ||
		    BeginQuery == nullptr ||
		    EndQuery == nullptr ||
		    GetQueryiv == nullptr ||
		    GetQueryObjectiv == nullptr ||
		    GetQueryObjectuiv == nullptr ||
		    BindBuffer == nullptr ||
		    DeleteBuffers == nullptr ||
		    GenBuffers == nullptr ||
		    IsBuffer == nullptr ||
		    BufferData == nullptr ||
		    BufferSubData == nullptr ||
		    GetBufferSubData == nullptr ||
		    MapBuffer == nullptr ||
		    UnmapBuffer == nullptr ||
		    GetBufferParameteriv == nullptr ||
		    GetBufferPointerv == nullptr ||
		    BlendEquationSeparate == nullptr ||
		    DrawBuffers == nullptr ||
		    StencilOpSeparate == nullptr ||
		    StencilFuncSeparate == nullptr ||
		    StencilMaskSeparate == nullptr ||
		    AttachShader == nullptr ||
		    BindAttribLocation == nullptr ||
		    CompileShader == nullptr ||
		    CreateProgram == nullptr ||
		    CreateShader == nullptr ||
		    DeleteProgram == nullptr ||
		    DeleteShader == nullptr ||
		    DetachShader == nullptr ||
		    DisableVertexAttribArray == nullptr ||
		    EnableVertexAttribArray == nullptr ||
		    GetActiveAttrib == nullptr ||
		    GetActiveUniform == nullptr ||
		    GetAttachedShaders == nullptr ||
		    GetAttribLocation == nullptr ||
		    GetProgramiv == nullptr ||
		    GetProgramInfoLog == nullptr ||
		    GetShaderiv == nullptr ||
		    GetShaderInfoLog == nullptr ||
		    GetShaderSource == nullptr ||
		    GetUniformLocation == nullptr ||
		    GetUniformfv == nullptr ||
		    GetUniformiv == nullptr ||
		    GetVertexAttribdv == nullptr ||
		    GetVertexAttribfv == nullptr ||
		    GetVertexAttribiv == nullptr ||
		    GetVertexAttribPointerv == nullptr ||
		    IsProgram == nullptr ||
		    IsShader == nullptr ||
		    LinkProgram == nullptr ||
		    ShaderSource == nullptr ||
		    UseProgram == nullptr ||
		    Uniform1f == nullptr ||
		    Uniform2f == nullptr ||
		    Uniform3f == nullptr ||
		    Uniform4f == nullptr ||
		    Uniform1i == nullptr ||
		    Uniform2i == nullptr ||
		    Uniform3i == nullptr ||
		    Uniform4i == nullptr ||
		    Uniform1fv == nullptr ||
		    Uniform2fv == nullptr ||
		    Uniform3fv == nullptr ||
		    Uniform4fv == nullptr ||
		    Uniform1iv == nullptr ||
		    Uniform2iv == nullptr ||
		    Uniform3iv == nullptr ||
		    Uniform4iv == nullptr ||
		    UniformMatrix2fv == nullptr ||
		    UniformMatrix3fv == nullptr ||
		    UniformMatrix4fv == nullptr ||
		    ValidateProgram == nullptr ||
		    VertexAttrib1d == nullptr ||
		    VertexAttrib1dv == nullptr ||
		    VertexAttrib1f == nullptr ||
		    VertexAttrib1fv == nullptr ||
		    VertexAttrib1s == nullptr ||
		    VertexAttrib1sv == nullptr ||
		    VertexAttrib2d == nullptr ||
		    VertexAttrib2dv == nullptr ||
		    VertexAttrib2f == nullptr ||
		    VertexAttrib2fv == nullptr ||
		    VertexAttrib2s == nullptr ||
		    VertexAttrib2sv == nullptr ||
		    VertexAttrib3d == nullptr ||
		    VertexAttrib3dv == nullptr ||
		    VertexAttrib3f == nullptr ||
		    VertexAttrib3fv == nullptr ||
		    VertexAttrib3s == nullptr ||
		    VertexAttrib3sv == nullptr ||
		    VertexAttrib4Nbv == nullptr ||
		    VertexAttrib4Niv == nullptr ||
		    VertexAttrib4Nsv == nullptr ||
		    VertexAttrib4Nub == nullptr ||
		    VertexAttrib4Nubv == nullptr ||
		    VertexAttrib4Nuiv == nullptr ||
		    VertexAttrib4Nusv == nullptr ||
		    VertexAttrib4bv == nullptr ||
		    VertexAttrib4d == nullptr ||
		    VertexAttrib4dv == nullptr ||
		    VertexAttrib4f == nullptr ||
		    VertexAttrib4fv == nullptr ||
		    VertexAttrib4iv == nullptr ||
		    VertexAttrib4s == nullptr ||
		    VertexAttrib4sv == nullptr ||
		    VertexAttrib4ubv == nullptr ||
		    VertexAttrib4uiv == nullptr ||
		    VertexAttrib4usv == nullptr ||
		    VertexAttribPointer == nullptr ||
		    UniformMatrix2x3fv == nullptr ||
		    UniformMatrix3x2fv == nullptr ||
		    UniformMatrix2x4fv == nullptr ||
		    UniformMatrix4x2fv == nullptr ||
		    UniformMatrix3x4fv == nullptr ||
		    UniformMatrix4x3fv == nullptr ||
		    ColorMaski == nullptr ||
		    GetBooleani_v == nullptr ||
		    GetIntegeri_v == nullptr ||
		    Enablei == nullptr ||
		    Disablei == nullptr ||
		    IsEnabledi == nullptr ||
		    BeginTransformFeedback == nullptr ||
		    EndTransformFeedback == nullptr ||
		    BindBufferRange == nullptr ||
		    BindBufferBase == nullptr ||
		    TransformFeedbackVaryings == nullptr ||
		    GetTransformFeedbackVarying == nullptr ||
		    ClampColor == nullptr ||
		    BeginConditionalRender == nullptr ||
		    EndConditionalRender == nullptr ||
		    VertexAttribIPointer == nullptr ||
		    GetVertexAttribIiv == nullptr ||
		    GetVertexAttribIuiv == nullptr ||
		    VertexAttribI1i == nullptr ||
		    VertexAttribI2i == nullptr ||
		    VertexAttribI3i == nullptr ||
		    VertexAttribI4i == nullptr ||
		    VertexAttribI1ui == nullptr ||
		    VertexAttribI2ui == nullptr ||
		    VertexAttribI3ui == nullptr ||
		    VertexAttribI4ui == nullptr ||
		    VertexAttribI1iv == nullptr ||
		    VertexAttribI2iv == nullptr ||
		    VertexAttribI3iv == nullptr ||
		    VertexAttribI4iv == nullptr ||
		    VertexAttribI1uiv == nullptr ||
		    VertexAttribI2uiv == nullptr ||
		    VertexAttribI3uiv == nullptr ||
		    VertexAttribI4uiv == nullptr ||
		    VertexAttribI4bv == nullptr ||
		    VertexAttribI4sv == nullptr ||
		    VertexAttribI4ubv == nullptr ||
		    VertexAttribI4usv == nullptr ||
		    GetUniformuiv == nullptr ||
		    BindFragDataLocation == nullptr ||
		    GetFragDataLocation == nullptr ||
		    Uniform1ui == nullptr ||
		    Uniform2ui == nullptr ||
		    Uniform3ui == nullptr ||
		    Uniform4ui == nullptr ||
		    Uniform1uiv == nullptr ||
		    Uniform2uiv == nullptr ||
		    Uniform3uiv == nullptr ||
		    Uniform4uiv == nullptr ||
		    TexParameterIiv == nullptr ||
		    TexParameterIuiv == nullptr ||
		    GetTexParameterIiv == nullptr ||
		    GetTexParameterIuiv == nullptr ||
		    ClearBufferiv == nullptr ||
		    ClearBufferuiv == nullptr ||
		    ClearBufferfv == nullptr ||
		    ClearBufferfi == nullptr ||
		    GetStringi == nullptr ||
		    IsRenderbuffer == nullptr ||
		    BindRenderbuffer == nullptr ||
		    DeleteRenderbuffers == nullptr ||
		    GenRenderbuffers == nullptr ||
		    RenderbufferStorage == nullptr ||
		    GetRenderbufferParameteriv == nullptr ||
		    IsFramebuffer == nullptr ||
		    BindFramebuffer == nullptr ||
		    DeleteFramebuffers == nullptr ||
		    GenFramebuffers == nullptr ||
		    CheckFramebufferStatus == nullptr ||
		    FramebufferTexture1D == nullptr ||
		    FramebufferTexture2D == nullptr ||
		    FramebufferTexture3D == nullptr ||
		    FramebufferRenderbuffer == nullptr ||
		    GetFramebufferAttachmentParameteriv == nullptr ||
		    GenerateMipmap == nullptr ||
		    BlitFramebuffer == nullptr ||
		    RenderbufferStorageMultisample == nullptr ||
		    FramebufferTextureLayer == nullptr ||
		    MapBufferRange == nullptr ||
		    FlushMappedBufferRange == nullptr ||
		    BindVertexArray == nullptr ||
		    DeleteVertexArrays == nullptr ||
		    GenVertexArrays == nullptr ||
		    IsVertexArray == nullptr ||
		    DrawArraysInstanced == nullptr ||
		    DrawElementsInstanced == nullptr ||
		    TexBuffer == nullptr ||
		    PrimitiveRestartIndex == nullptr ||
		    CopyBufferSubData == nullptr ||
		    GetUniformIndices == nullptr ||
		    GetActiveUniformsiv == nullptr ||
		    GetActiveUniformName == nullptr ||
		    GetUniformBlockIndex == nullptr ||
		    GetActiveUniformBlockiv == nullptr ||
		    GetActiveUniformBlockName == nullptr ||
		    UniformBlockBinding == nullptr ||
		    DrawElementsBaseVertex == nullptr ||
		    DrawRangeElementsBaseVertex == nullptr ||
		    DrawElementsInstancedBaseVertex == nullptr ||
		    MultiDrawElementsBaseVertex == nullptr ||
		    ProvokingVertex == nullptr ||
		    FenceSync == nullptr ||
		    IsSync == nullptr ||
		    DeleteSync == nullptr ||
		    ClientWaitSync == nullptr ||
		    WaitSync == nullptr ||
		    GetInteger64v == nullptr ||
		    GetSynciv == nullptr ||
		    GetInteger64i_v == nullptr ||
		    GetBufferParameteri64v == nullptr ||
		    FramebufferTexture == nullptr ||
		    TexImage2DMultisample == nullptr ||
		    TexImage3DMultisample == nullptr ||
		    GetMultisamplefv == nullptr ||
		    SampleMaski == nullptr ||
		    BindFragDataLocationIndexed == nullptr ||
		    GetFragDataIndex == nullptr ||
		    GenSamplers == nullptr ||
		    DeleteSamplers == nullptr ||
		    IsSampler == nullptr ||
		    BindSampler == nullptr ||
		    SamplerParameteri == nullptr ||
		    SamplerParameteriv == nullptr ||
		    SamplerParameterf == nullptr ||
		    SamplerParameterfv == nullptr ||
		    SamplerParameterIiv == nullptr ||
		    SamplerParameterIuiv == nullptr ||
		    GetSamplerParameteriv == nullptr ||
		    GetSamplerParameterIiv == nullptr ||
		    GetSamplerParameterfv == nullptr ||
		    GetSamplerParameterIuiv == nullptr ||
		    QueryCounter == nullptr ||
		    GetQueryObjecti64v == nullptr ||
		    GetQueryObjectui64v == nullptr ||
		    VertexAttribDivisor == nullptr ||
		    VertexAttribP1ui == nullptr ||
		    VertexAttribP1uiv == nullptr ||
		    VertexAttribP2ui == nullptr ||
		    VertexAttribP2uiv == nullptr ||
		    VertexAttribP3ui == nullptr ||
		    VertexAttribP3uiv == nullptr ||
		    VertexAttribP4ui == nullptr ||
		    VertexAttribP4uiv == nullptr ||
		    MinSampleShading == nullptr ||
		    BlendEquationi == nullptr ||
		    BlendEquationSeparatei == nullptr ||
		    BlendFunci == nullptr ||
		    BlendFuncSeparatei == nullptr ||
		    DrawArraysIndirect == nullptr ||
		    DrawElementsIndirect == nullptr ||
		    Uniform1d == nullptr ||
		    Uniform2d == nullptr ||
		    Uniform3d == nullptr ||
		    Uniform4d == nullptr ||
		    Uniform1dv == nullptr ||
		    Uniform2dv == nullptr ||
		    Uniform3dv == nullptr ||
		    Uniform4dv == nullptr ||
		    UniformMatrix2dv == nullptr ||
		    UniformMatrix3dv == nullptr ||
		    UniformMatrix4dv == nullptr ||
		    UniformMatrix2x3dv == nullptr ||
		    UniformMatrix2x4dv == nullptr ||
		    UniformMatrix3x2dv == nullptr ||
		    UniformMatrix3x4dv == nullptr ||
		    UniformMatrix4x2dv == nullptr ||
		    UniformMatrix4x3dv == nullptr ||
		    GetUniformdv == nullptr ||
		    GetSubroutineUniformLocation == nullptr ||
		    GetSubroutineIndex == nullptr ||
		    GetActiveSubroutineUniformiv == nullptr ||
		    GetActiveSubroutineUniformName == nullptr ||
		    GetActiveSubroutineName == nullptr ||
		    UniformSubroutinesuiv == nullptr ||
		    GetUniformSubroutineuiv == nullptr ||
		    GetProgramStageiv == nullptr ||
		    PatchParameteri == nullptr ||
		    PatchParameterfv == nullptr ||
		    BindTransformFeedback == nullptr ||
		    DeleteTransformFeedbacks == nullptr ||
		    GenTransformFeedbacks == nullptr ||
		    IsTransformFeedback == nullptr ||
		    PauseTransformFeedback == nullptr ||
		    ResumeTransformFeedback == nullptr ||
		    DrawTransformFeedback == nullptr ||
		    DrawTransformFeedbackStream == nullptr ||
		    BeginQueryIndexed == nullptr ||
		    EndQueryIndexed == nullptr ||
		    GetQueryIndexediv == nullptr ||
		    ReleaseShaderCompiler == nullptr ||
		    ShaderBinary == nullptr ||
		    GetShaderPrecisionFormat == nullptr ||
		    DepthRangef == nullptr ||
		    ClearDepthf == nullptr ||
		    GetProgramBinary == nullptr ||
		    ProgramBinary == nullptr ||
		    ProgramParameteri == nullptr ||
		    UseProgramStages == nullptr ||
		    ActiveShaderProgram == nullptr ||
		    CreateShaderProgramv == nullptr ||
		    BindProgramPipeline == nullptr ||
		    DeleteProgramPipelines == nullptr ||
		    GenProgramPipelines == nullptr ||
		    IsProgramPipeline == nullptr ||
		    GetProgramPipelineiv == nullptr ||
		    ProgramUniform1i == nullptr ||
		    ProgramUniform1iv == nullptr ||
		    ProgramUniform1f == nullptr ||
		    ProgramUniform1fv == nullptr ||
		    ProgramUniform1d == nullptr ||
		    ProgramUniform1dv == nullptr ||
		    ProgramUniform1ui == nullptr ||
		    ProgramUniform1uiv == nullptr ||
		    ProgramUniform2i == nullptr ||
		    ProgramUniform2iv == nullptr ||
		    ProgramUniform2f == nullptr ||
		    ProgramUniform2fv == nullptr ||
		    ProgramUniform2d == nullptr ||
		    ProgramUniform2dv == nullptr ||
		    ProgramUniform2ui == nullptr ||
		    ProgramUniform2uiv == nullptr ||
		    ProgramUniform3i == nullptr ||
		    ProgramUniform3iv == nullptr ||
		    ProgramUniform3f == nullptr ||
		    ProgramUniform3fv == nullptr ||
		    ProgramUniform3d == nullptr ||
		    ProgramUniform3dv == nullptr ||
		    ProgramUniform3ui == nullptr ||
		    ProgramUniform3uiv == nullptr ||
		    ProgramUniform4i == nullptr ||
		    ProgramUniform4iv == nullptr ||
		    ProgramUniform4f == nullptr ||
		    ProgramUniform4fv == nullptr ||
		    ProgramUniform4d == nullptr ||
		    ProgramUniform4dv == nullptr ||
		    ProgramUniform4ui == nullptr ||
		    ProgramUniform4uiv == nullptr ||
		    ProgramUniformMatrix2fv == nullptr ||
		    ProgramUniformMatrix3fv == nullptr ||
		    ProgramUniformMatrix4fv == nullptr ||
		    ProgramUniformMatrix2dv == nullptr ||
		    ProgramUniformMatrix3dv == nullptr ||
		    ProgramUniformMatrix4dv == nullptr ||
		    ProgramUniformMatrix2x3fv == nullptr ||
		    ProgramUniformMatrix3x2fv == nullptr ||
		    ProgramUniformMatrix2x4fv == nullptr ||
		    ProgramUniformMatrix4x2fv == nullptr ||
		    ProgramUniformMatrix3x4fv == nullptr ||
		    ProgramUniformMatrix4x3fv == nullptr ||
		    ProgramUniformMatrix2x3dv == nullptr ||
		    ProgramUniformMatrix3x2dv == nullptr ||
		    ProgramUniformMatrix2x4dv == nullptr ||
		    ProgramUniformMatrix4x2dv == nullptr ||
		    ProgramUniformMatrix3x4dv == nullptr ||
		    ProgramUniformMatrix4x3dv == nullptr ||
		    ValidateProgramPipeline == nullptr ||
		    GetProgramPipelineInfoLog == nullptr ||
		    VertexAttribL1d == nullptr ||
		    VertexAttribL2d == nullptr ||
		    VertexAttribL3d == nullptr ||
		    VertexAttribL4d == nullptr ||
		    VertexAttribL1dv == nullptr ||
		    VertexAttribL2dv == nullptr ||
		    VertexAttribL3dv == nullptr ||
		    VertexAttribL4dv == nullptr ||
		    VertexAttribLPointer == nullptr ||
		    GetVertexAttribLdv == nullptr ||
		    ViewportArrayv == nullptr ||
		    ViewportIndexedf == nullptr ||
		    ViewportIndexedfv == nullptr ||
		    ScissorArrayv == nullptr ||
		    ScissorIndexed == nullptr ||
		    ScissorIndexedv == nullptr ||
		    DepthRangeArrayv == nullptr ||
		    DepthRangeIndexed == nullptr ||
		    GetFloati_v == nullptr ||
		    GetDoublei_v == nullptr ||
		    DrawArraysInstancedBaseInstance == nullptr ||
		    DrawElementsInstancedBaseInstance == nullptr ||
		    DrawElementsInstancedBaseVertexBaseInstance == nullptr ||
		    GetInternalformativ == nullptr ||
		    GetActiveAtomicCounterBufferiv == nullptr ||
		    BindImageTexture == nullptr ||
		    MemoryBarrier == nullptr ||
		    TexStorage1D == nullptr ||
		    TexStorage2D == nullptr ||
		    TexStorage3D == nullptr ||
		    DrawTransformFeedbackInstanced == nullptr ||
		    DrawTransformFeedbackStreamInstanced == nullptr ||
		    ClearBufferData == nullptr ||
		    ClearBufferSubData == nullptr ||
		    DispatchCompute == nullptr ||
		    DispatchComputeIndirect == nullptr ||
		    CopyImageSubData == nullptr ||
		    FramebufferParameteri == nullptr ||
		    GetFramebufferParameteriv == nullptr ||
		    GetInternalformati64v == nullptr ||
		    InvalidateTexSubImage == nullptr ||
		    InvalidateTexImage == nullptr ||
		    InvalidateBufferSubData == nullptr ||
		    InvalidateBufferData == nullptr ||
		    InvalidateFramebuffer == nullptr ||
		    InvalidateSubFramebuffer == nullptr ||
		    MultiDrawArraysIndirect == nullptr ||
		    MultiDrawElementsIndirect == nullptr ||
		    GetProgramInterfaceiv == nullptr ||
		    GetProgramResourceIndex == nullptr ||
		    GetProgramResourceName == nullptr ||
		    GetProgramResourceiv == nullptr ||
		    GetProgramResourceLocation == nullptr ||
		    GetProgramResourceLocationIndex == nullptr ||
		    ShaderStorageBlockBinding == nullptr ||
		    TexBufferRange == nullptr ||
		    TexStorage2DMultisample == nullptr ||
		    TexStorage3DMultisample == nullptr ||
		    TextureView == nullptr ||
		    BindVertexBuffer == nullptr ||
		    VertexAttribFormat == nullptr ||
		    VertexAttribIFormat == nullptr ||
		    VertexAttribLFormat == nullptr ||
		    VertexAttribBinding == nullptr ||
		    VertexBindingDivisor == nullptr ||
		    DebugMessageControl == nullptr ||
		    DebugMessageInsert == nullptr ||
		    DebugMessageCallback == nullptr ||
		    GetDebugMessageLog == nullptr ||
		    PushDebugGroup == nullptr ||
		    PopDebugGroup == nullptr ||
		    ObjectLabel == nullptr ||
		    GetObjectLabel == nullptr ||
		    ObjectPtrLabel == nullptr ||
		    GetObjectPtrLabel == nullptr ||
		    BufferStorage == nullptr ||
		    ClearTexImage == nullptr ||
		    ClearTexSubImage == nullptr ||
		    BindBuffersBase == nullptr ||
		    BindBuffersRange == nullptr ||
		    BindTextures == nullptr ||
		    BindSamplers == nullptr ||
		    BindImageTextures == nullptr ||
		    BindVertexBuffers == nullptr ||
		    ClipControl == nullptr ||
		    CreateTransformFeedbacks == nullptr ||
		    TransformFeedbackBufferBase == nullptr ||
		    TransformFeedbackBufferRange == nullptr ||
		    GetTransformFeedbackiv == nullptr ||
		    GetTransformFeedbacki_v == nullptr ||
		    GetTransformFeedbacki64_v == nullptr ||
		    CreateBuffers == nullptr ||
		    NamedBufferStorage == nullptr ||
		    NamedBufferData == nullptr ||
		    NamedBufferSubData == nullptr ||
		    CopyNamedBufferSubData == nullptr ||
		    ClearNamedBufferData == nullptr ||
		    ClearNamedBufferSubData == nullptr ||
		    MapNamedBuffer == nullptr ||
		    MapNamedBufferRange == nullptr ||
		    UnmapNamedBuffer == nullptr ||
		    FlushMappedNamedBufferRange == nullptr ||
		    GetNamedBufferParameteriv == nullptr ||
		    GetNamedBufferParameteri64v == nullptr ||
		    GetNamedBufferPointerv == nullptr ||
		    GetNamedBufferSubData == nullptr ||
		    CreateFramebuffers == nullptr ||
		    NamedFramebufferRenderbuffer == nullptr ||
		    NamedFramebufferParameteri == nullptr ||
		    NamedFramebufferTexture == nullptr ||
		    NamedFramebufferTextureLayer == nullptr ||
		    NamedFramebufferDrawBuffer == nullptr ||
		    NamedFramebufferDrawBuffers == nullptr ||
		    NamedFramebufferReadBuffer == nullptr ||
		    InvalidateNamedFramebufferData == nullptr ||
		    InvalidateNamedFramebufferSubData == nullptr ||
		    ClearNamedFramebufferiv == nullptr ||
		    ClearNamedFramebufferuiv == nullptr ||
		    ClearNamedFramebufferfv == nullptr ||
		    ClearNamedFramebufferfi == nullptr ||
		    BlitNamedFramebuffer == nullptr ||
		    CheckNamedFramebufferStatus == nullptr ||
		    GetNamedFramebufferParameteriv == nullptr ||
		    GetNamedFramebufferAttachmentParameteriv == nullptr ||
		    CreateRenderbuffers == nullptr ||
		    NamedRenderbufferStorage == nullptr ||
		    NamedRenderbufferStorageMultisample == nullptr ||
		    GetNamedRenderbufferParameteriv == nullptr ||
		    CreateTextures == nullptr ||
		    TextureBuffer == nullptr ||
		    TextureBufferRange == nullptr ||
		    TextureStorage1D == nullptr ||
		    TextureStorage2D == nullptr ||
		    TextureStorage3D == nullptr ||
		    TextureStorage2DMultisample == nullptr ||
		    TextureStorage3DMultisample == nullptr ||
		    TextureSubImage1D == nullptr ||
		    TextureSubImage2D == nullptr ||
		    TextureSubImage3D == nullptr ||
		    CompressedTextureSubImage1D == nullptr ||
		    CompressedTextureSubImage2D == nullptr ||
		    CompressedTextureSubImage3D == nullptr ||
		    CopyTextureSubImage1D == nullptr ||
		    CopyTextureSubImage2D == nullptr ||
		    CopyTextureSubImage3D == nullptr ||
		    TextureParameterf == nullptr ||
		    TextureParameterfv == nullptr ||
		    TextureParameteri == nullptr ||
		    TextureParameterIiv == nullptr ||
		    TextureParameterIuiv == nullptr ||
		    TextureParameteriv == nullptr ||
		    GenerateTextureMipmap == nullptr ||
		    BindTextureUnit == nullptr ||
		    GetTextureImage == nullptr ||
		    GetCompressedTextureImage == nullptr ||
		    GetTextureLevelParameterfv == nullptr ||
		    GetTextureLevelParameteriv == nullptr ||
		    GetTextureParameterfv == nullptr ||
		    GetTextureParameterIiv == nullptr ||
		    GetTextureParameterIuiv == nullptr ||
		    GetTextureParameteriv == nullptr ||
		    CreateVertexArrays == nullptr ||
		    DisableVertexArrayAttrib == nullptr ||
		    EnableVertexArrayAttrib == nullptr ||
		    VertexArrayElementBuffer == nullptr ||
		    VertexArrayVertexBuffer == nullptr ||
		    VertexArrayVertexBuffers == nullptr ||
		    VertexArrayAttribBinding == nullptr ||
		    VertexArrayAttribFormat == nullptr ||
		    VertexArrayAttribIFormat == nullptr ||
		    VertexArrayAttribLFormat == nullptr ||
		    VertexArrayBindingDivisor == nullptr ||
		    GetVertexArrayiv == nullptr ||
		    GetVertexArrayIndexediv == nullptr ||
		    GetVertexArrayIndexed64iv == nullptr ||
		    CreateSamplers == nullptr ||
		    CreateProgramPipelines == nullptr ||
		    CreateQueries == nullptr ||
		    GetQueryBufferObjecti64v == nullptr ||
		    GetQueryBufferObjectiv == nullptr ||
		    GetQueryBufferObjectui64v == nullptr ||
		    GetQueryBufferObjectuiv == nullptr ||
		    MemoryBarrierByRegion == nullptr ||
		    GetTextureSubImage == nullptr ||
		    GetCompressedTextureSubImage == nullptr ||
		    GetGraphicsResetStatus == nullptr ||
		    GetnCompressedTexImage == nullptr ||
		    GetnTexImage == nullptr ||
		    GetnUniformdv == nullptr ||
		    GetnUniformfv == nullptr ||
		    GetnUniformiv == nullptr ||
		    GetnUniformuiv == nullptr ||
		    ReadnPixels == nullptr ||
		    TextureBarrier == nullptr)
			return; // throw std::runtime_error("OpenGL IAT initialization failed");
	}

	const glcoreContext* glcoreContextInit()
	{
		return new glcoreContext;
	}
	
	void glcoreContextDestroy(const glcoreContext* ctx)
	{
		delete ctx;
	}
	
	void glcoreContextMakeCurrent(const glcoreContext* ctx)
	{
		context = ctx;
	}
	
	const glcoreContext* glcoreContextGetCurrent()
	{
		return context;
	}


	void glActiveShaderProgram(GLuint pipeline, GLuint program)
	{
		return context->ActiveShaderProgram(pipeline, program);
	}

	void glActiveTexture(GLenum texture)
	{
		return context->ActiveTexture(texture);
	}

	void glAttachShader(GLuint program, GLuint shader)
	{
		return context->AttachShader(program, shader);
	}

	void glBeginConditionalRender(GLuint id, GLenum mode)
	{
		return context->BeginConditionalRender(id, mode);
	}

	void glBeginQuery(GLenum target, GLuint id)
	{
		return context->BeginQuery(target, id);
	}

	void glBeginQueryIndexed(GLenum target, GLuint index, GLuint id)
	{
		return context->BeginQueryIndexed(target, index, id);
	}

	void glBeginTransformFeedback(GLenum primitiveMode)
	{
		return context->BeginTransformFeedback(primitiveMode);
	}

	void glBindAttribLocation(GLuint program, GLuint index, const GLchar* name)
	{
		return context->BindAttribLocation(program, index, name);
	}

	void glBindBuffer(GLenum target, GLuint buffer)
	{
		return context->BindBuffer(target, buffer);
	}

	void glBindBufferBase(GLenum target, GLuint index, GLuint buffer)
	{
		return context->BindBufferBase(target, index, buffer);
	}

	void glBindBufferRange(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size)
	{
		return context->BindBufferRange(target, index, buffer, offset, size);
	}

	void glBindBuffersBase(GLenum target, GLuint first, GLsizei count, const GLuint* buffers)
	{
		return context->BindBuffersBase(target, first, count, buffers);
	}

	void glBindBuffersRange(GLenum target, GLuint first, GLsizei count, const GLuint* buffers, const GLintptr* offsets, const GLsizeiptr* sizes)
	{
		return context->BindBuffersRange(target, first, count, buffers, offsets, sizes);
	}

	void glBindFragDataLocation(GLuint program, GLuint color, const GLchar* name)
	{
		return context->BindFragDataLocation(program, color, name);
	}

	void glBindFragDataLocationIndexed(GLuint program, GLuint colorNumber, GLuint index, const GLchar* name)
	{
		return context->BindFragDataLocationIndexed(program, colorNumber, index, name);
	}

	void glBindFramebuffer(GLenum target, GLuint framebuffer)
	{
		return context->BindFramebuffer(target, framebuffer);
	}

	void glBindImageTexture(GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format)
	{
		return context->BindImageTexture(unit, texture, level, layered, layer, access, format);
	}

	void glBindImageTextures(GLuint first, GLsizei count, const GLuint* textures)
	{
		return context->BindImageTextures(first, count, textures);
	}

	void glBindProgramPipeline(GLuint pipeline)
	{
		return context->BindProgramPipeline(pipeline);
	}

	void glBindRenderbuffer(GLenum target, GLuint renderbuffer)
	{
		return context->BindRenderbuffer(target, renderbuffer);
	}

	void glBindSampler(GLuint unit, GLuint sampler)
	{
		return context->BindSampler(unit, sampler);
	}

	void glBindSamplers(GLuint first, GLsizei count, const GLuint* samplers)
	{
		return context->BindSamplers(first, count, samplers);
	}

	void glBindTexture(GLenum target, GLuint texture)
	{
		return context->BindTexture(target, texture);
	}

	void glBindTextureUnit(GLuint unit, GLuint texture)
	{
		return context->BindTextureUnit(unit, texture);
	}

	void glBindTextures(GLuint first, GLsizei count, const GLuint* textures)
	{
		return context->BindTextures(first, count, textures);
	}

	void glBindTransformFeedback(GLenum target, GLuint id)
	{
		return context->BindTransformFeedback(target, id);
	}

	void glBindVertexArray(GLuint array)
	{
		return context->BindVertexArray(array);
	}

	void glBindVertexBuffer(GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride)
	{
		return context->BindVertexBuffer(bindingindex, buffer, offset, stride);
	}

	void glBindVertexBuffers(GLuint first, GLsizei count, const GLuint* buffers, const GLintptr* offsets, const GLsizei* strides)
	{
		return context->BindVertexBuffers(first, count, buffers, offsets, strides);
	}

	void glBlendColor(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
	{
		return context->BlendColor(red, green, blue, alpha);
	}

	void glBlendEquation(GLenum mode)
	{
		return context->BlendEquation(mode);
	}

	void glBlendEquationSeparate(GLenum modeRGB, GLenum modeAlpha)
	{
		return context->BlendEquationSeparate(modeRGB, modeAlpha);
	}

	void glBlendEquationSeparatei(GLuint buf, GLenum modeRGB, GLenum modeAlpha)
	{
		return context->BlendEquationSeparatei(buf, modeRGB, modeAlpha);
	}

	void glBlendEquationi(GLuint buf, GLenum mode)
	{
		return context->BlendEquationi(buf, mode);
	}

	void glBlendFunc(GLenum sfactor, GLenum dfactor)
	{
		return context->BlendFunc(sfactor, dfactor);
	}

	void glBlendFuncSeparate(GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha)
	{
		return context->BlendFuncSeparate(sfactorRGB, dfactorRGB, sfactorAlpha, dfactorAlpha);
	}

	void glBlendFuncSeparatei(GLuint buf, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha)
	{
		return context->BlendFuncSeparatei(buf, srcRGB, dstRGB, srcAlpha, dstAlpha);
	}

	void glBlendFunci(GLuint buf, GLenum src, GLenum dst)
	{
		return context->BlendFunci(buf, src, dst);
	}

	void glBlitFramebuffer(GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter)
	{
		return context->BlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
	}

	void glBlitNamedFramebuffer(GLuint readFramebuffer, GLuint drawFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter)
	{
		return context->BlitNamedFramebuffer(readFramebuffer, drawFramebuffer, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
	}

	void glBufferData(GLenum target, GLsizeiptr size, const void* data, GLenum usage)
	{
		return context->BufferData(target, size, data, usage);
	}

	void glBufferStorage(GLenum target, GLsizeiptr size, const void* data, GLbitfield flags)
	{
		return context->BufferStorage(target, size, data, flags);
	}

	void glBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, const void* data)
	{
		return context->BufferSubData(target, offset, size, data);
	}

	GLenum glCheckFramebufferStatus(GLenum target)
	{
		return context->CheckFramebufferStatus(target);
	}

	GLenum glCheckNamedFramebufferStatus(GLuint framebuffer, GLenum target)
	{
		return context->CheckNamedFramebufferStatus(framebuffer, target);
	}

	void glClampColor(GLenum target, GLenum clamp)
	{
		return context->ClampColor(target, clamp);
	}

	void glClear(GLbitfield mask)
	{
		return context->Clear(mask);
	}

	void glClearBufferData(GLenum target, GLenum internalformat, GLenum format, GLenum type, const void* data)
	{
		return context->ClearBufferData(target, internalformat, format, type, data);
	}

	void glClearBufferSubData(GLenum target, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void* data)
	{
		return context->ClearBufferSubData(target, internalformat, offset, size, format, type, data);
	}

	void glClearBufferfi(GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil)
	{
		return context->ClearBufferfi(buffer, drawbuffer, depth, stencil);
	}

	void glClearBufferfv(GLenum buffer, GLint drawbuffer, const GLfloat* value)
	{
		return context->ClearBufferfv(buffer, drawbuffer, value);
	}

	void glClearBufferiv(GLenum buffer, GLint drawbuffer, const GLint* value)
	{
		return context->ClearBufferiv(buffer, drawbuffer, value);
	}

	void glClearBufferuiv(GLenum buffer, GLint drawbuffer, const GLuint* value)
	{
		return context->ClearBufferuiv(buffer, drawbuffer, value);
	}

	void glClearColor(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
	{
		return context->ClearColor(red, green, blue, alpha);
	}

	void glClearDepth(GLdouble depth)
	{
		return context->ClearDepth(depth);
	}

	void glClearDepthf(GLfloat d)
	{
		return context->ClearDepthf(d);
	}

	void glClearNamedBufferData(GLuint buffer, GLenum internalformat, GLenum format, GLenum type, const void* data)
	{
		return context->ClearNamedBufferData(buffer, internalformat, format, type, data);
	}

	void glClearNamedBufferSubData(GLuint buffer, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void* data)
	{
		return context->ClearNamedBufferSubData(buffer, internalformat, offset, size, format, type, data);
	}

	void glClearNamedFramebufferfi(GLuint framebuffer, GLenum buffer, const GLfloat depth, GLint stencil)
	{
		return context->ClearNamedFramebufferfi(framebuffer, buffer, depth, stencil);
	}

	void glClearNamedFramebufferfv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLfloat* value)
	{
		return context->ClearNamedFramebufferfv(framebuffer, buffer, drawbuffer, value);
	}

	void glClearNamedFramebufferiv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLint* value)
	{
		return context->ClearNamedFramebufferiv(framebuffer, buffer, drawbuffer, value);
	}

	void glClearNamedFramebufferuiv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLuint* value)
	{
		return context->ClearNamedFramebufferuiv(framebuffer, buffer, drawbuffer, value);
	}

	void glClearStencil(GLint s)
	{
		return context->ClearStencil(s);
	}

	void glClearTexImage(GLuint texture, GLint level, GLenum format, GLenum type, const void* data)
	{
		return context->ClearTexImage(texture, level, format, type, data);
	}

	void glClearTexSubImage(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* data)
	{
		return context->ClearTexSubImage(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, type, data);
	}

	GLenum glClientWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout)
	{
		return context->ClientWaitSync(sync, flags, timeout);
	}

	void glClipControl(GLenum origin, GLenum depth)
	{
		return context->ClipControl(origin, depth);
	}

	void glColorMask(GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha)
	{
		return context->ColorMask(red, green, blue, alpha);
	}

	void glColorMaski(GLuint index, GLboolean r, GLboolean g, GLboolean b, GLboolean a)
	{
		return context->ColorMaski(index, r, g, b, a);
	}

	void glCompileShader(GLuint shader)
	{
		return context->CompileShader(shader);
	}

	void glCompressedTexImage1D(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void* data)
	{
		return context->CompressedTexImage1D(target, level, internalformat, width, border, imageSize, data);
	}

	void glCompressedTexImage2D(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void* data)
	{
		return context->CompressedTexImage2D(target, level, internalformat, width, height, border, imageSize, data);
	}

	void glCompressedTexImage3D(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void* data)
	{
		return context->CompressedTexImage3D(target, level, internalformat, width, height, depth, border, imageSize, data);
	}

	void glCompressedTexSubImage1D(GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void* data)
	{
		return context->CompressedTexSubImage1D(target, level, xoffset, width, format, imageSize, data);
	}

	void glCompressedTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void* data)
	{
		return context->CompressedTexSubImage2D(target, level, xoffset, yoffset, width, height, format, imageSize, data);
	}

	void glCompressedTexSubImage3D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void* data)
	{
		return context->CompressedTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
	}

	void glCompressedTextureSubImage1D(GLuint texture, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void* data)
	{
		return context->CompressedTextureSubImage1D(texture, level, xoffset, width, format, imageSize, data);
	}

	void glCompressedTextureSubImage2D(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void* data)
	{
		return context->CompressedTextureSubImage2D(texture, level, xoffset, yoffset, width, height, format, imageSize, data);
	}

	void glCompressedTextureSubImage3D(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void* data)
	{
		return context->CompressedTextureSubImage3D(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
	}

	void glCopyBufferSubData(GLenum readTarget, GLenum writeTarget, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size)
	{
		return context->CopyBufferSubData(readTarget, writeTarget, readOffset, writeOffset, size);
	}

	void glCopyImageSubData(GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, GLuint dstName, GLenum dstTarget, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei srcWidth, GLsizei srcHeight, GLsizei srcDepth)
	{
		return context->CopyImageSubData(srcName, srcTarget, srcLevel, srcX, srcY, srcZ, dstName, dstTarget, dstLevel, dstX, dstY, dstZ, srcWidth, srcHeight, srcDepth);
	}

	void glCopyNamedBufferSubData(GLuint readBuffer, GLuint writeBuffer, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size)
	{
		return context->CopyNamedBufferSubData(readBuffer, writeBuffer, readOffset, writeOffset, size);
	}

	void glCopyTexImage1D(GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border)
	{
		return context->CopyTexImage1D(target, level, internalformat, x, y, width, border);
	}

	void glCopyTexImage2D(GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border)
	{
		return context->CopyTexImage2D(target, level, internalformat, x, y, width, height, border);
	}

	void glCopyTexSubImage1D(GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width)
	{
		return context->CopyTexSubImage1D(target, level, xoffset, x, y, width);
	}

	void glCopyTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height)
	{
		return context->CopyTexSubImage2D(target, level, xoffset, yoffset, x, y, width, height);
	}

	void glCopyTexSubImage3D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height)
	{
		return context->CopyTexSubImage3D(target, level, xoffset, yoffset, zoffset, x, y, width, height);
	}

	void glCopyTextureSubImage1D(GLuint texture, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width)
	{
		return context->CopyTextureSubImage1D(texture, level, xoffset, x, y, width);
	}

	void glCopyTextureSubImage2D(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height)
	{
		return context->CopyTextureSubImage2D(texture, level, xoffset, yoffset, x, y, width, height);
	}

	void glCopyTextureSubImage3D(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height)
	{
		return context->CopyTextureSubImage3D(texture, level, xoffset, yoffset, zoffset, x, y, width, height);
	}

	void glCreateBuffers(GLsizei n, GLuint* buffers)
	{
		return context->CreateBuffers(n, buffers);
	}

	void glCreateFramebuffers(GLsizei n, GLuint* framebuffers)
	{
		return context->CreateFramebuffers(n, framebuffers);
	}

	GLuint glCreateProgram()
	{
		return context->CreateProgram();
	}

	void glCreateProgramPipelines(GLsizei n, GLuint* pipelines)
	{
		return context->CreateProgramPipelines(n, pipelines);
	}

	void glCreateQueries(GLenum target, GLsizei n, GLuint* ids)
	{
		return context->CreateQueries(target, n, ids);
	}

	void glCreateRenderbuffers(GLsizei n, GLuint* renderbuffers)
	{
		return context->CreateRenderbuffers(n, renderbuffers);
	}

	void glCreateSamplers(GLsizei n, GLuint* samplers)
	{
		return context->CreateSamplers(n, samplers);
	}

	GLuint glCreateShader(GLenum type)
	{
		return context->CreateShader(type);
	}

	GLuint glCreateShaderProgramv(GLenum type, GLsizei count, const GLchar* const* strings)
	{
		return context->CreateShaderProgramv(type, count, strings);
	}

	void glCreateTextures(GLenum target, GLsizei n, GLuint* textures)
	{
		return context->CreateTextures(target, n, textures);
	}

	void glCreateTransformFeedbacks(GLsizei n, GLuint* ids)
	{
		return context->CreateTransformFeedbacks(n, ids);
	}

	void glCreateVertexArrays(GLsizei n, GLuint* arrays)
	{
		return context->CreateVertexArrays(n, arrays);
	}

	void glCullFace(GLenum mode)
	{
		return context->CullFace(mode);
	}

	void glDebugMessageCallback(GLDEBUGPROC callback, const void* userParam)
	{
		return context->DebugMessageCallback(callback, userParam);
	}

	void glDebugMessageControl(GLenum source, GLenum type, GLenum severity, GLsizei count, const GLuint* ids, GLboolean enabled)
	{
		return context->DebugMessageControl(source, type, severity, count, ids, enabled);
	}

	void glDebugMessageInsert(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* buf)
	{
		return context->DebugMessageInsert(source, type, id, severity, length, buf);
	}

	void glDeleteBuffers(GLsizei n, const GLuint* buffers)
	{
		return context->DeleteBuffers(n, buffers);
	}

	void glDeleteFramebuffers(GLsizei n, const GLuint* framebuffers)
	{
		return context->DeleteFramebuffers(n, framebuffers);
	}

	void glDeleteProgram(GLuint program)
	{
		return context->DeleteProgram(program);
	}

	void glDeleteProgramPipelines(GLsizei n, const GLuint* pipelines)
	{
		return context->DeleteProgramPipelines(n, pipelines);
	}

	void glDeleteQueries(GLsizei n, const GLuint* ids)
	{
		return context->DeleteQueries(n, ids);
	}

	void glDeleteRenderbuffers(GLsizei n, const GLuint* renderbuffers)
	{
		return context->DeleteRenderbuffers(n, renderbuffers);
	}

	void glDeleteSamplers(GLsizei count, const GLuint* samplers)
	{
		return context->DeleteSamplers(count, samplers);
	}

	void glDeleteShader(GLuint shader)
	{
		return context->DeleteShader(shader);
	}

	void glDeleteSync(GLsync sync)
	{
		return context->DeleteSync(sync);
	}

	void glDeleteTextures(GLsizei n, const GLuint* textures)
	{
		return context->DeleteTextures(n, textures);
	}

	void glDeleteTransformFeedbacks(GLsizei n, const GLuint* ids)
	{
		return context->DeleteTransformFeedbacks(n, ids);
	}

	void glDeleteVertexArrays(GLsizei n, const GLuint* arrays)
	{
		return context->DeleteVertexArrays(n, arrays);
	}

	void glDepthFunc(GLenum func)
	{
		return context->DepthFunc(func);
	}

	void glDepthMask(GLboolean flag)
	{
		return context->DepthMask(flag);
	}

	void glDepthRange(GLdouble near, GLdouble far)
	{
		return context->DepthRange(near, far);
	}

	void glDepthRangeArrayv(GLuint first, GLsizei count, const GLdouble* v)
	{
		return context->DepthRangeArrayv(first, count, v);
	}

	void glDepthRangeIndexed(GLuint index, GLdouble n, GLdouble f)
	{
		return context->DepthRangeIndexed(index, n, f);
	}

	void glDepthRangef(GLfloat n, GLfloat f)
	{
		return context->DepthRangef(n, f);
	}

	void glDetachShader(GLuint program, GLuint shader)
	{
		return context->DetachShader(program, shader);
	}

	void glDisable(GLenum cap)
	{
		return context->Disable(cap);
	}

	void glDisableVertexArrayAttrib(GLuint vaobj, GLuint index)
	{
		return context->DisableVertexArrayAttrib(vaobj, index);
	}

	void glDisableVertexAttribArray(GLuint index)
	{
		return context->DisableVertexAttribArray(index);
	}

	void glDisablei(GLenum target, GLuint index)
	{
		return context->Disablei(target, index);
	}

	void glDispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z)
	{
		return context->DispatchCompute(num_groups_x, num_groups_y, num_groups_z);
	}

	void glDispatchComputeIndirect(GLintptr indirect)
	{
		return context->DispatchComputeIndirect(indirect);
	}

	void glDrawArrays(GLenum mode, GLint first, GLsizei count)
	{
		return context->DrawArrays(mode, first, count);
	}

	void glDrawArraysIndirect(GLenum mode, const void* indirect)
	{
		return context->DrawArraysIndirect(mode, indirect);
	}

	void glDrawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instancecount)
	{
		return context->DrawArraysInstanced(mode, first, count, instancecount);
	}

	void glDrawArraysInstancedBaseInstance(GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance)
	{
		return context->DrawArraysInstancedBaseInstance(mode, first, count, instancecount, baseinstance);
	}

	void glDrawBuffer(GLenum buf)
	{
		return context->DrawBuffer(buf);
	}

	void glDrawBuffers(GLsizei n, const GLenum* bufs)
	{
		return context->DrawBuffers(n, bufs);
	}

	void glDrawElements(GLenum mode, GLsizei count, GLenum type, const void* indices)
	{
		return context->DrawElements(mode, count, type, indices);
	}

	void glDrawElementsBaseVertex(GLenum mode, GLsizei count, GLenum type, const void* indices, GLint basevertex)
	{
		return context->DrawElementsBaseVertex(mode, count, type, indices, basevertex);
	}

	void glDrawElementsIndirect(GLenum mode, GLenum type, const void* indirect)
	{
		return context->DrawElementsIndirect(mode, type, indirect);
	}

	void glDrawElementsInstanced(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount)
	{
		return context->DrawElementsInstanced(mode, count, type, indices, instancecount);
	}

	void glDrawElementsInstancedBaseInstance(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount, GLuint baseinstance)
	{
		return context->DrawElementsInstancedBaseInstance(mode, count, type, indices, instancecount, baseinstance);
	}

	void glDrawElementsInstancedBaseVertex(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount, GLint basevertex)
	{
		return context->DrawElementsInstancedBaseVertex(mode, count, type, indices, instancecount, basevertex);
	}

	void glDrawElementsInstancedBaseVertexBaseInstance(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance)
	{
		return context->DrawElementsInstancedBaseVertexBaseInstance(mode, count, type, indices, instancecount, basevertex, baseinstance);
	}

	void glDrawRangeElements(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void* indices)
	{
		return context->DrawRangeElements(mode, start, end, count, type, indices);
	}

	void glDrawRangeElementsBaseVertex(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void* indices, GLint basevertex)
	{
		return context->DrawRangeElementsBaseVertex(mode, start, end, count, type, indices, basevertex);
	}

	void glDrawTransformFeedback(GLenum mode, GLuint id)
	{
		return context->DrawTransformFeedback(mode, id);
	}

	void glDrawTransformFeedbackInstanced(GLenum mode, GLuint id, GLsizei instancecount)
	{
		return context->DrawTransformFeedbackInstanced(mode, id, instancecount);
	}

	void glDrawTransformFeedbackStream(GLenum mode, GLuint id, GLuint stream)
	{
		return context->DrawTransformFeedbackStream(mode, id, stream);
	}

	void glDrawTransformFeedbackStreamInstanced(GLenum mode, GLuint id, GLuint stream, GLsizei instancecount)
	{
		return context->DrawTransformFeedbackStreamInstanced(mode, id, stream, instancecount);
	}

	void glEnable(GLenum cap)
	{
		return context->Enable(cap);
	}

	void glEnableVertexArrayAttrib(GLuint vaobj, GLuint index)
	{
		return context->EnableVertexArrayAttrib(vaobj, index);
	}

	void glEnableVertexAttribArray(GLuint index)
	{
		return context->EnableVertexAttribArray(index);
	}

	void glEnablei(GLenum target, GLuint index)
	{
		return context->Enablei(target, index);
	}

	void glEndConditionalRender()
	{
		return context->EndConditionalRender();
	}

	void glEndQuery(GLenum target)
	{
		return context->EndQuery(target);
	}

	void glEndQueryIndexed(GLenum target, GLuint index)
	{
		return context->EndQueryIndexed(target, index);
	}

	void glEndTransformFeedback()
	{
		return context->EndTransformFeedback();
	}

	GLsync glFenceSync(GLenum condition, GLbitfield flags)
	{
		return context->FenceSync(condition, flags);
	}

	void glFinish()
	{
		return context->Finish();
	}

	void glFlush()
	{
		return context->Flush();
	}

	void glFlushMappedBufferRange(GLenum target, GLintptr offset, GLsizeiptr length)
	{
		return context->FlushMappedBufferRange(target, offset, length);
	}

	void glFlushMappedNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length)
	{
		return context->FlushMappedNamedBufferRange(buffer, offset, length);
	}

	void glFramebufferParameteri(GLenum target, GLenum pname, GLint param)
	{
		return context->FramebufferParameteri(target, pname, param);
	}

	void glFramebufferRenderbuffer(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer)
	{
		return context->FramebufferRenderbuffer(target, attachment, renderbuffertarget, renderbuffer);
	}

	void glFramebufferTexture(GLenum target, GLenum attachment, GLuint texture, GLint level)
	{
		return context->FramebufferTexture(target, attachment, texture, level);
	}

	void glFramebufferTexture1D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level)
	{
		return context->FramebufferTexture1D(target, attachment, textarget, texture, level);
	}

	void glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level)
	{
		return context->FramebufferTexture2D(target, attachment, textarget, texture, level);
	}

	void glFramebufferTexture3D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint zoffset)
	{
		return context->FramebufferTexture3D(target, attachment, textarget, texture, level, zoffset);
	}

	void glFramebufferTextureLayer(GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer)
	{
		return context->FramebufferTextureLayer(target, attachment, texture, level, layer);
	}

	void glFrontFace(GLenum mode)
	{
		return context->FrontFace(mode);
	}

	void glGenBuffers(GLsizei n, GLuint* buffers)
	{
		return context->GenBuffers(n, buffers);
	}

	void glGenFramebuffers(GLsizei n, GLuint* framebuffers)
	{
		return context->GenFramebuffers(n, framebuffers);
	}

	void glGenProgramPipelines(GLsizei n, GLuint* pipelines)
	{
		return context->GenProgramPipelines(n, pipelines);
	}

	void glGenQueries(GLsizei n, GLuint* ids)
	{
		return context->GenQueries(n, ids);
	}

	void glGenRenderbuffers(GLsizei n, GLuint* renderbuffers)
	{
		return context->GenRenderbuffers(n, renderbuffers);
	}

	void glGenSamplers(GLsizei count, GLuint* samplers)
	{
		return context->GenSamplers(count, samplers);
	}

	void glGenTextures(GLsizei n, GLuint* textures)
	{
		return context->GenTextures(n, textures);
	}

	void glGenTransformFeedbacks(GLsizei n, GLuint* ids)
	{
		return context->GenTransformFeedbacks(n, ids);
	}

	void glGenVertexArrays(GLsizei n, GLuint* arrays)
	{
		return context->GenVertexArrays(n, arrays);
	}

	void glGenerateMipmap(GLenum target)
	{
		return context->GenerateMipmap(target);
	}

	void glGenerateTextureMipmap(GLuint texture)
	{
		return context->GenerateTextureMipmap(texture);
	}

	void glGetActiveAtomicCounterBufferiv(GLuint program, GLuint bufferIndex, GLenum pname, GLint* params)
	{
		return context->GetActiveAtomicCounterBufferiv(program, bufferIndex, pname, params);
	}

	void glGetActiveAttrib(GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLint* size, GLenum* type, GLchar* name)
	{
		return context->GetActiveAttrib(program, index, bufSize, length, size, type, name);
	}

	void glGetActiveSubroutineName(GLuint program, GLenum shadertype, GLuint index, GLsizei bufsize, GLsizei* length, GLchar* name)
	{
		return context->GetActiveSubroutineName(program, shadertype, index, bufsize, length, name);
	}

	void glGetActiveSubroutineUniformName(GLuint program, GLenum shadertype, GLuint index, GLsizei bufsize, GLsizei* length, GLchar* name)
	{
		return context->GetActiveSubroutineUniformName(program, shadertype, index, bufsize, length, name);
	}

	void glGetActiveSubroutineUniformiv(GLuint program, GLenum shadertype, GLuint index, GLenum pname, GLint* values)
	{
		return context->GetActiveSubroutineUniformiv(program, shadertype, index, pname, values);
	}

	void glGetActiveUniform(GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLint* size, GLenum* type, GLchar* name)
	{
		return context->GetActiveUniform(program, index, bufSize, length, size, type, name);
	}

	void glGetActiveUniformBlockName(GLuint program, GLuint uniformBlockIndex, GLsizei bufSize, GLsizei* length, GLchar* uniformBlockName)
	{
		return context->GetActiveUniformBlockName(program, uniformBlockIndex, bufSize, length, uniformBlockName);
	}

	void glGetActiveUniformBlockiv(GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint* params)
	{
		return context->GetActiveUniformBlockiv(program, uniformBlockIndex, pname, params);
	}

	void glGetActiveUniformName(GLuint program, GLuint uniformIndex, GLsizei bufSize, GLsizei* length, GLchar* uniformName)
	{
		return context->GetActiveUniformName(program, uniformIndex, bufSize, length, uniformName);
	}

	void glGetActiveUniformsiv(GLuint program, GLsizei uniformCount, const GLuint* uniformIndices, GLenum pname, GLint* params)
	{
		return context->GetActiveUniformsiv(program, uniformCount, uniformIndices, pname, params);
	}

	void glGetAttachedShaders(GLuint program, GLsizei maxCount, GLsizei* count, GLuint* shaders)
	{
		return context->GetAttachedShaders(program, maxCount, count, shaders);
	}

	GLint glGetAttribLocation(GLuint program, const GLchar* name)
	{
		return context->GetAttribLocation(program, name);
	}

	void glGetBooleani_v(GLenum target, GLuint index, GLboolean* data)
	{
		return context->GetBooleani_v(target, index, data);
	}

	void glGetBooleanv(GLenum pname, GLboolean* data)
	{
		return context->GetBooleanv(pname, data);
	}

	void glGetBufferParameteri64v(GLenum target, GLenum pname, GLint64* params)
	{
		return context->GetBufferParameteri64v(target, pname, params);
	}

	void glGetBufferParameteriv(GLenum target, GLenum pname, GLint* params)
	{
		return context->GetBufferParameteriv(target, pname, params);
	}

	void glGetBufferPointerv(GLenum target, GLenum pname, void** params)
	{
		return context->GetBufferPointerv(target, pname, params);
	}

	void glGetBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, void* data)
	{
		return context->GetBufferSubData(target, offset, size, data);
	}

	void glGetCompressedTexImage(GLenum target, GLint level, void* img)
	{
		return context->GetCompressedTexImage(target, level, img);
	}

	void glGetCompressedTextureImage(GLuint texture, GLint level, GLsizei bufSize, void* pixels)
	{
		return context->GetCompressedTextureImage(texture, level, bufSize, pixels);
	}

	void glGetCompressedTextureSubImage(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLsizei bufSize, void* pixels)
	{
		return context->GetCompressedTextureSubImage(texture, level, xoffset, yoffset, zoffset, width, height, depth, bufSize, pixels);
	}

	GLuint glGetDebugMessageLog(GLuint count, GLsizei bufSize, GLenum* sources, GLenum* types, GLuint* ids, GLenum* severities, GLsizei* lengths, GLchar* messageLog)
	{
		return context->GetDebugMessageLog(count, bufSize, sources, types, ids, severities, lengths, messageLog);
	}

	void glGetDoublei_v(GLenum target, GLuint index, GLdouble* data)
	{
		return context->GetDoublei_v(target, index, data);
	}

	void glGetDoublev(GLenum pname, GLdouble* data)
	{
		return context->GetDoublev(pname, data);
	}

	GLenum glGetError()
	{
		return context->GetError();
	}

	void glGetFloati_v(GLenum target, GLuint index, GLfloat* data)
	{
		return context->GetFloati_v(target, index, data);
	}

	void glGetFloatv(GLenum pname, GLfloat* data)
	{
		return context->GetFloatv(pname, data);
	}

	GLint glGetFragDataIndex(GLuint program, const GLchar* name)
	{
		return context->GetFragDataIndex(program, name);
	}

	GLint glGetFragDataLocation(GLuint program, const GLchar* name)
	{
		return context->GetFragDataLocation(program, name);
	}

	void glGetFramebufferAttachmentParameteriv(GLenum target, GLenum attachment, GLenum pname, GLint* params)
	{
		return context->GetFramebufferAttachmentParameteriv(target, attachment, pname, params);
	}

	void glGetFramebufferParameteriv(GLenum target, GLenum pname, GLint* params)
	{
		return context->GetFramebufferParameteriv(target, pname, params);
	}

	GLenum glGetGraphicsResetStatus()
	{
		return context->GetGraphicsResetStatus();
	}

	void glGetInteger64i_v(GLenum target, GLuint index, GLint64* data)
	{
		return context->GetInteger64i_v(target, index, data);
	}

	void glGetInteger64v(GLenum pname, GLint64* data)
	{
		return context->GetInteger64v(pname, data);
	}

	void glGetIntegeri_v(GLenum target, GLuint index, GLint* data)
	{
		return context->GetIntegeri_v(target, index, data);
	}

	void glGetIntegerv(GLenum pname, GLint* data)
	{
		return context->GetIntegerv(pname, data);
	}

	void glGetInternalformati64v(GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint64* params)
	{
		return context->GetInternalformati64v(target, internalformat, pname, bufSize, params);
	}

	void glGetInternalformativ(GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint* params)
	{
		return context->GetInternalformativ(target, internalformat, pname, bufSize, params);
	}

	void glGetMultisamplefv(GLenum pname, GLuint index, GLfloat* val)
	{
		return context->GetMultisamplefv(pname, index, val);
	}

	void glGetNamedBufferParameteri64v(GLuint buffer, GLenum pname, GLint64* params)
	{
		return context->GetNamedBufferParameteri64v(buffer, pname, params);
	}

	void glGetNamedBufferParameteriv(GLuint buffer, GLenum pname, GLint* params)
	{
		return context->GetNamedBufferParameteriv(buffer, pname, params);
	}

	void glGetNamedBufferPointerv(GLuint buffer, GLenum pname, void** params)
	{
		return context->GetNamedBufferPointerv(buffer, pname, params);
	}

	void glGetNamedBufferSubData(GLuint buffer, GLintptr offset, GLsizeiptr size, void* data)
	{
		return context->GetNamedBufferSubData(buffer, offset, size, data);
	}

	void glGetNamedFramebufferAttachmentParameteriv(GLuint framebuffer, GLenum attachment, GLenum pname, GLint* params)
	{
		return context->GetNamedFramebufferAttachmentParameteriv(framebuffer, attachment, pname, params);
	}

	void glGetNamedFramebufferParameteriv(GLuint framebuffer, GLenum pname, GLint* param)
	{
		return context->GetNamedFramebufferParameteriv(framebuffer, pname, param);
	}

	void glGetNamedRenderbufferParameteriv(GLuint renderbuffer, GLenum pname, GLint* params)
	{
		return context->GetNamedRenderbufferParameteriv(renderbuffer, pname, params);
	}

	void glGetObjectLabel(GLenum identifier, GLuint name, GLsizei bufSize, GLsizei* length, GLchar* label)
	{
		return context->GetObjectLabel(identifier, name, bufSize, length, label);
	}

	void glGetObjectPtrLabel(const void* ptr, GLsizei bufSize, GLsizei* length, GLchar* label)
	{
		return context->GetObjectPtrLabel(ptr, bufSize, length, label);
	}

	void glGetProgramBinary(GLuint program, GLsizei bufSize, GLsizei* length, GLenum* binaryFormat, void* binary)
	{
		return context->GetProgramBinary(program, bufSize, length, binaryFormat, binary);
	}

	void glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei* length, GLchar* infoLog)
	{
		return context->GetProgramInfoLog(program, bufSize, length, infoLog);
	}

	void glGetProgramInterfaceiv(GLuint program, GLenum programInterface, GLenum pname, GLint* params)
	{
		return context->GetProgramInterfaceiv(program, programInterface, pname, params);
	}

	void glGetProgramPipelineInfoLog(GLuint pipeline, GLsizei bufSize, GLsizei* length, GLchar* infoLog)
	{
		return context->GetProgramPipelineInfoLog(pipeline, bufSize, length, infoLog);
	}

	void glGetProgramPipelineiv(GLuint pipeline, GLenum pname, GLint* params)
	{
		return context->GetProgramPipelineiv(pipeline, pname, params);
	}

	GLuint glGetProgramResourceIndex(GLuint program, GLenum programInterface, const GLchar* name)
	{
		return context->GetProgramResourceIndex(program, programInterface, name);
	}

	GLint glGetProgramResourceLocation(GLuint program, GLenum programInterface, const GLchar* name)
	{
		return context->GetProgramResourceLocation(program, programInterface, name);
	}

	GLint glGetProgramResourceLocationIndex(GLuint program, GLenum programInterface, const GLchar* name)
	{
		return context->GetProgramResourceLocationIndex(program, programInterface, name);
	}

	void glGetProgramResourceName(GLuint program, GLenum programInterface, GLuint index, GLsizei bufSize, GLsizei* length, GLchar* name)
	{
		return context->GetProgramResourceName(program, programInterface, index, bufSize, length, name);
	}

	void glGetProgramResourceiv(GLuint program, GLenum programInterface, GLuint index, GLsizei propCount, const GLenum* props, GLsizei bufSize, GLsizei* length, GLint* params)
	{
		return context->GetProgramResourceiv(program, programInterface, index, propCount, props, bufSize, length, params);
	}

	void glGetProgramStageiv(GLuint program, GLenum shadertype, GLenum pname, GLint* values)
	{
		return context->GetProgramStageiv(program, shadertype, pname, values);
	}

	void glGetProgramiv(GLuint program, GLenum pname, GLint* params)
	{
		return context->GetProgramiv(program, pname, params);
	}

	void glGetQueryBufferObjecti64v(GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
	{
		return context->GetQueryBufferObjecti64v(id, buffer, pname, offset);
	}

	void glGetQueryBufferObjectiv(GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
	{
		return context->GetQueryBufferObjectiv(id, buffer, pname, offset);
	}

	void glGetQueryBufferObjectui64v(GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
	{
		return context->GetQueryBufferObjectui64v(id, buffer, pname, offset);
	}

	void glGetQueryBufferObjectuiv(GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
	{
		return context->GetQueryBufferObjectuiv(id, buffer, pname, offset);
	}

	void glGetQueryIndexediv(GLenum target, GLuint index, GLenum pname, GLint* params)
	{
		return context->GetQueryIndexediv(target, index, pname, params);
	}

	void glGetQueryObjecti64v(GLuint id, GLenum pname, GLint64* params)
	{
		return context->GetQueryObjecti64v(id, pname, params);
	}

	void glGetQueryObjectiv(GLuint id, GLenum pname, GLint* params)
	{
		return context->GetQueryObjectiv(id, pname, params);
	}

	void glGetQueryObjectui64v(GLuint id, GLenum pname, GLuint64* params)
	{
		return context->GetQueryObjectui64v(id, pname, params);
	}

	void glGetQueryObjectuiv(GLuint id, GLenum pname, GLuint* params)
	{
		return context->GetQueryObjectuiv(id, pname, params);
	}

	void glGetQueryiv(GLenum target, GLenum pname, GLint* params)
	{
		return context->GetQueryiv(target, pname, params);
	}

	void glGetRenderbufferParameteriv(GLenum target, GLenum pname, GLint* params)
	{
		return context->GetRenderbufferParameteriv(target, pname, params);
	}

	void glGetSamplerParameterIiv(GLuint sampler, GLenum pname, GLint* params)
	{
		return context->GetSamplerParameterIiv(sampler, pname, params);
	}

	void glGetSamplerParameterIuiv(GLuint sampler, GLenum pname, GLuint* params)
	{
		return context->GetSamplerParameterIuiv(sampler, pname, params);
	}

	void glGetSamplerParameterfv(GLuint sampler, GLenum pname, GLfloat* params)
	{
		return context->GetSamplerParameterfv(sampler, pname, params);
	}

	void glGetSamplerParameteriv(GLuint sampler, GLenum pname, GLint* params)
	{
		return context->GetSamplerParameteriv(sampler, pname, params);
	}

	void glGetShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* infoLog)
	{
		return context->GetShaderInfoLog(shader, bufSize, length, infoLog);
	}

	void glGetShaderPrecisionFormat(GLenum shadertype, GLenum precisiontype, GLint* range, GLint* precision)
	{
		return context->GetShaderPrecisionFormat(shadertype, precisiontype, range, precision);
	}

	void glGetShaderSource(GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* source)
	{
		return context->GetShaderSource(shader, bufSize, length, source);
	}

	void glGetShaderiv(GLuint shader, GLenum pname, GLint* params)
	{
		return context->GetShaderiv(shader, pname, params);
	}

	const GLubyte* glGetString(GLenum name)
	{
		return context->GetString(name);
	}

	const GLubyte* glGetStringi(GLenum name, GLuint index)
	{
		return context->GetStringi(name, index);
	}

	GLuint glGetSubroutineIndex(GLuint program, GLenum shadertype, const GLchar* name)
	{
		return context->GetSubroutineIndex(program, shadertype, name);
	}

	GLint glGetSubroutineUniformLocation(GLuint program, GLenum shadertype, const GLchar* name)
	{
		return context->GetSubroutineUniformLocation(program, shadertype, name);
	}

	void glGetSynciv(GLsync sync, GLenum pname, GLsizei bufSize, GLsizei* length, GLint* values)
	{
		return context->GetSynciv(sync, pname, bufSize, length, values);
	}

	void glGetTexImage(GLenum target, GLint level, GLenum format, GLenum type, void* pixels)
	{
		return context->GetTexImage(target, level, format, type, pixels);
	}

	void glGetTexLevelParameterfv(GLenum target, GLint level, GLenum pname, GLfloat* params)
	{
		return context->GetTexLevelParameterfv(target, level, pname, params);
	}

	void glGetTexLevelParameteriv(GLenum target, GLint level, GLenum pname, GLint* params)
	{
		return context->GetTexLevelParameteriv(target, level, pname, params);
	}

	void glGetTexParameterIiv(GLenum target, GLenum pname, GLint* params)
	{
		return context->GetTexParameterIiv(target, pname, params);
	}

	void glGetTexParameterIuiv(GLenum target, GLenum pname, GLuint* params)
	{
		return context->GetTexParameterIuiv(target, pname, params);
	}

	void glGetTexParameterfv(GLenum target, GLenum pname, GLfloat* params)
	{
		return context->GetTexParameterfv(target, pname, params);
	}

	void glGetTexParameteriv(GLenum target, GLenum pname, GLint* params)
	{
		return context->GetTexParameteriv(target, pname, params);
	}

	void glGetTextureImage(GLuint texture, GLint level, GLenum format, GLenum type, GLsizei bufSize, void* pixels)
	{
		return context->GetTextureImage(texture, level, format, type, bufSize, pixels);
	}

	void glGetTextureLevelParameterfv(GLuint texture, GLint level, GLenum pname, GLfloat* params)
	{
		return context->GetTextureLevelParameterfv(texture, level, pname, params);
	}

	void glGetTextureLevelParameteriv(GLuint texture, GLint level, GLenum pname, GLint* params)
	{
		return context->GetTextureLevelParameteriv(texture, level, pname, params);
	}

	void glGetTextureParameterIiv(GLuint texture, GLenum pname, GLint* params)
	{
		return context->GetTextureParameterIiv(texture, pname, params);
	}

	void glGetTextureParameterIuiv(GLuint texture, GLenum pname, GLuint* params)
	{
		return context->GetTextureParameterIuiv(texture, pname, params);
	}

	void glGetTextureParameterfv(GLuint texture, GLenum pname, GLfloat* params)
	{
		return context->GetTextureParameterfv(texture, pname, params);
	}

	void glGetTextureParameteriv(GLuint texture, GLenum pname, GLint* params)
	{
		return context->GetTextureParameteriv(texture, pname, params);
	}

	void glGetTextureSubImage(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLsizei bufSize, void* pixels)
	{
		return context->GetTextureSubImage(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, type, bufSize, pixels);
	}

	void glGetTransformFeedbackVarying(GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLsizei* size, GLenum* type, GLchar* name)
	{
		return context->GetTransformFeedbackVarying(program, index, bufSize, length, size, type, name);
	}

	void glGetTransformFeedbacki64_v(GLuint xfb, GLenum pname, GLuint index, GLint64* param)
	{
		return context->GetTransformFeedbacki64_v(xfb, pname, index, param);
	}

	void glGetTransformFeedbacki_v(GLuint xfb, GLenum pname, GLuint index, GLint* param)
	{
		return context->GetTransformFeedbacki_v(xfb, pname, index, param);
	}

	void glGetTransformFeedbackiv(GLuint xfb, GLenum pname, GLint* param)
	{
		return context->GetTransformFeedbackiv(xfb, pname, param);
	}

	GLuint glGetUniformBlockIndex(GLuint program, const GLchar* uniformBlockName)
	{
		return context->GetUniformBlockIndex(program, uniformBlockName);
	}

	void glGetUniformIndices(GLuint program, GLsizei uniformCount, const GLchar* const* uniformNames, GLuint* uniformIndices)
	{
		return context->GetUniformIndices(program, uniformCount, uniformNames, uniformIndices);
	}

	GLint glGetUniformLocation(GLuint program, const GLchar* name)
	{
		return context->GetUniformLocation(program, name);
	}

	void glGetUniformSubroutineuiv(GLenum shadertype, GLint location, GLuint* params)
	{
		return context->GetUniformSubroutineuiv(shadertype, location, params);
	}

	void glGetUniformdv(GLuint program, GLint location, GLdouble* params)
	{
		return context->GetUniformdv(program, location, params);
	}

	void glGetUniformfv(GLuint program, GLint location, GLfloat* params)
	{
		return context->GetUniformfv(program, location, params);
	}

	void glGetUniformiv(GLuint program, GLint location, GLint* params)
	{
		return context->GetUniformiv(program, location, params);
	}

	void glGetUniformuiv(GLuint program, GLint location, GLuint* params)
	{
		return context->GetUniformuiv(program, location, params);
	}

	void glGetVertexArrayIndexed64iv(GLuint vaobj, GLuint index, GLenum pname, GLint64* param)
	{
		return context->GetVertexArrayIndexed64iv(vaobj, index, pname, param);
	}

	void glGetVertexArrayIndexediv(GLuint vaobj, GLuint index, GLenum pname, GLint* param)
	{
		return context->GetVertexArrayIndexediv(vaobj, index, pname, param);
	}

	void glGetVertexArrayiv(GLuint vaobj, GLenum pname, GLint* param)
	{
		return context->GetVertexArrayiv(vaobj, pname, param);
	}

	void glGetVertexAttribIiv(GLuint index, GLenum pname, GLint* params)
	{
		return context->GetVertexAttribIiv(index, pname, params);
	}

	void glGetVertexAttribIuiv(GLuint index, GLenum pname, GLuint* params)
	{
		return context->GetVertexAttribIuiv(index, pname, params);
	}

	void glGetVertexAttribLdv(GLuint index, GLenum pname, GLdouble* params)
	{
		return context->GetVertexAttribLdv(index, pname, params);
	}

	void glGetVertexAttribPointerv(GLuint index, GLenum pname, void** pointer)
	{
		return context->GetVertexAttribPointerv(index, pname, pointer);
	}

	void glGetVertexAttribdv(GLuint index, GLenum pname, GLdouble* params)
	{
		return context->GetVertexAttribdv(index, pname, params);
	}

	void glGetVertexAttribfv(GLuint index, GLenum pname, GLfloat* params)
	{
		return context->GetVertexAttribfv(index, pname, params);
	}

	void glGetVertexAttribiv(GLuint index, GLenum pname, GLint* params)
	{
		return context->GetVertexAttribiv(index, pname, params);
	}

	void glGetnCompressedTexImage(GLenum target, GLint lod, GLsizei bufSize, void* pixels)
	{
		return context->GetnCompressedTexImage(target, lod, bufSize, pixels);
	}

	void glGetnTexImage(GLenum target, GLint level, GLenum format, GLenum type, GLsizei bufSize, void* pixels)
	{
		return context->GetnTexImage(target, level, format, type, bufSize, pixels);
	}

	void glGetnUniformdv(GLuint program, GLint location, GLsizei bufSize, GLdouble* params)
	{
		return context->GetnUniformdv(program, location, bufSize, params);
	}

	void glGetnUniformfv(GLuint program, GLint location, GLsizei bufSize, GLfloat* params)
	{
		return context->GetnUniformfv(program, location, bufSize, params);
	}

	void glGetnUniformiv(GLuint program, GLint location, GLsizei bufSize, GLint* params)
	{
		return context->GetnUniformiv(program, location, bufSize, params);
	}

	void glGetnUniformuiv(GLuint program, GLint location, GLsizei bufSize, GLuint* params)
	{
		return context->GetnUniformuiv(program, location, bufSize, params);
	}

	void glHint(GLenum target, GLenum mode)
	{
		return context->Hint(target, mode);
	}

	void glInvalidateBufferData(GLuint buffer)
	{
		return context->InvalidateBufferData(buffer);
	}

	void glInvalidateBufferSubData(GLuint buffer, GLintptr offset, GLsizeiptr length)
	{
		return context->InvalidateBufferSubData(buffer, offset, length);
	}

	void glInvalidateFramebuffer(GLenum target, GLsizei numAttachments, const GLenum* attachments)
	{
		return context->InvalidateFramebuffer(target, numAttachments, attachments);
	}

	void glInvalidateNamedFramebufferData(GLuint framebuffer, GLsizei numAttachments, const GLenum* attachments)
	{
		return context->InvalidateNamedFramebufferData(framebuffer, numAttachments, attachments);
	}

	void glInvalidateNamedFramebufferSubData(GLuint framebuffer, GLsizei numAttachments, const GLenum* attachments, GLint x, GLint y, GLsizei width, GLsizei height)
	{
		return context->InvalidateNamedFramebufferSubData(framebuffer, numAttachments, attachments, x, y, width, height);
	}

	void glInvalidateSubFramebuffer(GLenum target, GLsizei numAttachments, const GLenum* attachments, GLint x, GLint y, GLsizei width, GLsizei height)
	{
		return context->InvalidateSubFramebuffer(target, numAttachments, attachments, x, y, width, height);
	}

	void glInvalidateTexImage(GLuint texture, GLint level)
	{
		return context->InvalidateTexImage(texture, level);
	}

	void glInvalidateTexSubImage(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth)
	{
		return context->InvalidateTexSubImage(texture, level, xoffset, yoffset, zoffset, width, height, depth);
	}

	GLboolean glIsBuffer(GLuint buffer)
	{
		return context->IsBuffer(buffer);
	}

	GLboolean glIsEnabled(GLenum cap)
	{
		return context->IsEnabled(cap);
	}

	GLboolean glIsEnabledi(GLenum target, GLuint index)
	{
		return context->IsEnabledi(target, index);
	}

	GLboolean glIsFramebuffer(GLuint framebuffer)
	{
		return context->IsFramebuffer(framebuffer);
	}

	GLboolean glIsProgram(GLuint program)
	{
		return context->IsProgram(program);
	}

	GLboolean glIsProgramPipeline(GLuint pipeline)
	{
		return context->IsProgramPipeline(pipeline);
	}

	GLboolean glIsQuery(GLuint id)
	{
		return context->IsQuery(id);
	}

	GLboolean glIsRenderbuffer(GLuint renderbuffer)
	{
		return context->IsRenderbuffer(renderbuffer);
	}

	GLboolean glIsSampler(GLuint sampler)
	{
		return context->IsSampler(sampler);
	}

	GLboolean glIsShader(GLuint shader)
	{
		return context->IsShader(shader);
	}

	GLboolean glIsSync(GLsync sync)
	{
		return context->IsSync(sync);
	}

	GLboolean glIsTexture(GLuint texture)
	{
		return context->IsTexture(texture);
	}

	GLboolean glIsTransformFeedback(GLuint id)
	{
		return context->IsTransformFeedback(id);
	}

	GLboolean glIsVertexArray(GLuint array)
	{
		return context->IsVertexArray(array);
	}

	void glLineWidth(GLfloat width)
	{
		return context->LineWidth(width);
	}

	void glLinkProgram(GLuint program)
	{
		return context->LinkProgram(program);
	}

	void glLogicOp(GLenum opcode)
	{
		return context->LogicOp(opcode);
	}

	void* glMapBuffer(GLenum target, GLenum access)
	{
		return context->MapBuffer(target, access);
	}

	void* glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access)
	{
		return context->MapBufferRange(target, offset, length, access);
	}

	void* glMapNamedBuffer(GLuint buffer, GLenum access)
	{
		return context->MapNamedBuffer(buffer, access);
	}

	void* glMapNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield access)
	{
		return context->MapNamedBufferRange(buffer, offset, length, access);
	}

	void glMemoryBarrier(GLbitfield barriers)
	{
		return context->MemoryBarrier(barriers);
	}

	void glMemoryBarrierByRegion(GLbitfield barriers)
	{
		return context->MemoryBarrierByRegion(barriers);
	}

	void glMinSampleShading(GLfloat value)
	{
		return context->MinSampleShading(value);
	}

	void glMultiDrawArrays(GLenum mode, const GLint* first, const GLsizei* count, GLsizei drawcount)
	{
		return context->MultiDrawArrays(mode, first, count, drawcount);
	}

	void glMultiDrawArraysIndirect(GLenum mode, const void* indirect, GLsizei drawcount, GLsizei stride)
	{
		return context->MultiDrawArraysIndirect(mode, indirect, drawcount, stride);
	}

	void glMultiDrawElements(GLenum mode, const GLsizei* count, GLenum type, const void* const* indices, GLsizei drawcount)
	{
		return context->MultiDrawElements(mode, count, type, indices, drawcount);
	}

	void glMultiDrawElementsBaseVertex(GLenum mode, const GLsizei* count, GLenum type, const void* const* indices, GLsizei drawcount, const GLint* basevertex)
	{
		return context->MultiDrawElementsBaseVertex(mode, count, type, indices, drawcount, basevertex);
	}

	void glMultiDrawElementsIndirect(GLenum mode, GLenum type, const void* indirect, GLsizei drawcount, GLsizei stride)
	{
		return context->MultiDrawElementsIndirect(mode, type, indirect, drawcount, stride);
	}

	void glNamedBufferData(GLuint buffer, GLsizeiptr size, const void* data, GLenum usage)
	{
		return context->NamedBufferData(buffer, size, data, usage);
	}

	void glNamedBufferStorage(GLuint buffer, GLsizeiptr size, const void* data, GLbitfield flags)
	{
		return context->NamedBufferStorage(buffer, size, data, flags);
	}

	void glNamedBufferSubData(GLuint buffer, GLintptr offset, GLsizeiptr size, const void* data)
	{
		return context->NamedBufferSubData(buffer, offset, size, data);
	}

	void glNamedFramebufferDrawBuffer(GLuint framebuffer, GLenum buf)
	{
		return context->NamedFramebufferDrawBuffer(framebuffer, buf);
	}

	void glNamedFramebufferDrawBuffers(GLuint framebuffer, GLsizei n, const GLenum* bufs)
	{
		return context->NamedFramebufferDrawBuffers(framebuffer, n, bufs);
	}

	void glNamedFramebufferParameteri(GLuint framebuffer, GLenum pname, GLint param)
	{
		return context->NamedFramebufferParameteri(framebuffer, pname, param);
	}

	void glNamedFramebufferReadBuffer(GLuint framebuffer, GLenum src)
	{
		return context->NamedFramebufferReadBuffer(framebuffer, src);
	}

	void glNamedFramebufferRenderbuffer(GLuint framebuffer, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer)
	{
		return context->NamedFramebufferRenderbuffer(framebuffer, attachment, renderbuffertarget, renderbuffer);
	}

	void glNamedFramebufferTexture(GLuint framebuffer, GLenum attachment, GLuint texture, GLint level)
	{
		return context->NamedFramebufferTexture(framebuffer, attachment, texture, level);
	}

	void glNamedFramebufferTextureLayer(GLuint framebuffer, GLenum attachment, GLuint texture, GLint level, GLint layer)
	{
		return context->NamedFramebufferTextureLayer(framebuffer, attachment, texture, level, layer);
	}

	void glNamedRenderbufferStorage(GLuint renderbuffer, GLenum internalformat, GLsizei width, GLsizei height)
	{
		return context->NamedRenderbufferStorage(renderbuffer, internalformat, width, height);
	}

	void glNamedRenderbufferStorageMultisample(GLuint renderbuffer, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height)
	{
		return context->NamedRenderbufferStorageMultisample(renderbuffer, samples, internalformat, width, height);
	}

	void glObjectLabel(GLenum identifier, GLuint name, GLsizei length, const GLchar* label)
	{
		return context->ObjectLabel(identifier, name, length, label);
	}

	void glObjectPtrLabel(const void* ptr, GLsizei length, const GLchar* label)
	{
		return context->ObjectPtrLabel(ptr, length, label);
	}

	void glPatchParameterfv(GLenum pname, const GLfloat* values)
	{
		return context->PatchParameterfv(pname, values);
	}

	void glPatchParameteri(GLenum pname, GLint value)
	{
		return context->PatchParameteri(pname, value);
	}

	void glPauseTransformFeedback()
	{
		return context->PauseTransformFeedback();
	}

	void glPixelStoref(GLenum pname, GLfloat param)
	{
		return context->PixelStoref(pname, param);
	}

	void glPixelStorei(GLenum pname, GLint param)
	{
		return context->PixelStorei(pname, param);
	}

	void glPointParameterf(GLenum pname, GLfloat param)
	{
		return context->PointParameterf(pname, param);
	}

	void glPointParameterfv(GLenum pname, const GLfloat* params)
	{
		return context->PointParameterfv(pname, params);
	}

	void glPointParameteri(GLenum pname, GLint param)
	{
		return context->PointParameteri(pname, param);
	}

	void glPointParameteriv(GLenum pname, const GLint* params)
	{
		return context->PointParameteriv(pname, params);
	}

	void glPointSize(GLfloat size)
	{
		return context->PointSize(size);
	}

	void glPolygonMode(GLenum face, GLenum mode)
	{
		return context->PolygonMode(face, mode);
	}

	void glPolygonOffset(GLfloat factor, GLfloat units)
	{
		return context->PolygonOffset(factor, units);
	}

	void glPopDebugGroup()
	{
		return context->PopDebugGroup();
	}

	void glPrimitiveRestartIndex(GLuint index)
	{
		return context->PrimitiveRestartIndex(index);
	}

	void glProgramBinary(GLuint program, GLenum binaryFormat, const void* binary, GLsizei length)
	{
		return context->ProgramBinary(program, binaryFormat, binary, length);
	}

	void glProgramParameteri(GLuint program, GLenum pname, GLint value)
	{
		return context->ProgramParameteri(program, pname, value);
	}

	void glProgramUniform1d(GLuint program, GLint location, GLdouble v0)
	{
		return context->ProgramUniform1d(program, location, v0);
	}

	void glProgramUniform1dv(GLuint program, GLint location, GLsizei count, const GLdouble* value)
	{
		return context->ProgramUniform1dv(program, location, count, value);
	}

	void glProgramUniform1f(GLuint program, GLint location, GLfloat v0)
	{
		return context->ProgramUniform1f(program, location, v0);
	}

	void glProgramUniform1fv(GLuint program, GLint location, GLsizei count, const GLfloat* value)
	{
		return context->ProgramUniform1fv(program, location, count, value);
	}

	void glProgramUniform1i(GLuint program, GLint location, GLint v0)
	{
		return context->ProgramUniform1i(program, location, v0);
	}

	void glProgramUniform1iv(GLuint program, GLint location, GLsizei count, const GLint* value)
	{
		return context->ProgramUniform1iv(program, location, count, value);
	}

	void glProgramUniform1ui(GLuint program, GLint location, GLuint v0)
	{
		return context->ProgramUniform1ui(program, location, v0);
	}

	void glProgramUniform1uiv(GLuint program, GLint location, GLsizei count, const GLuint* value)
	{
		return context->ProgramUniform1uiv(program, location, count, value);
	}

	void glProgramUniform2d(GLuint program, GLint location, GLdouble v0, GLdouble v1)
	{
		return context->ProgramUniform2d(program, location, v0, v1);
	}

	void glProgramUniform2dv(GLuint program, GLint location, GLsizei count, const GLdouble* value)
	{
		return context->ProgramUniform2dv(program, location, count, value);
	}

	void glProgramUniform2f(GLuint program, GLint location, GLfloat v0, GLfloat v1)
	{
		return context->ProgramUniform2f(program, location, v0, v1);
	}

	void glProgramUniform2fv(GLuint program, GLint location, GLsizei count, const GLfloat* value)
	{
		return context->ProgramUniform2fv(program, location, count, value);
	}

	void glProgramUniform2i(GLuint program, GLint location, GLint v0, GLint v1)
	{
		return context->ProgramUniform2i(program, location, v0, v1);
	}

	void glProgramUniform2iv(GLuint program, GLint location, GLsizei count, const GLint* value)
	{
		return context->ProgramUniform2iv(program, location, count, value);
	}

	void glProgramUniform2ui(GLuint program, GLint location, GLuint v0, GLuint v1)
	{
		return context->ProgramUniform2ui(program, location, v0, v1);
	}

	void glProgramUniform2uiv(GLuint program, GLint location, GLsizei count, const GLuint* value)
	{
		return context->ProgramUniform2uiv(program, location, count, value);
	}

	void glProgramUniform3d(GLuint program, GLint location, GLdouble v0, GLdouble v1, GLdouble v2)
	{
		return context->ProgramUniform3d(program, location, v0, v1, v2);
	}

	void glProgramUniform3dv(GLuint program, GLint location, GLsizei count, const GLdouble* value)
	{
		return context->ProgramUniform3dv(program, location, count, value);
	}

	void glProgramUniform3f(GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2)
	{
		return context->ProgramUniform3f(program, location, v0, v1, v2);
	}

	void glProgramUniform3fv(GLuint program, GLint location, GLsizei count, const GLfloat* value)
	{
		return context->ProgramUniform3fv(program, location, count, value);
	}

	void glProgramUniform3i(GLuint program, GLint location, GLint v0, GLint v1, GLint v2)
	{
		return context->ProgramUniform3i(program, location, v0, v1, v2);
	}

	void glProgramUniform3iv(GLuint program, GLint location, GLsizei count, const GLint* value)
	{
		return context->ProgramUniform3iv(program, location, count, value);
	}

	void glProgramUniform3ui(GLuint program, GLint location, GLuint v0, GLuint v1, GLuint v2)
	{
		return context->ProgramUniform3ui(program, location, v0, v1, v2);
	}

	void glProgramUniform3uiv(GLuint program, GLint location, GLsizei count, const GLuint* value)
	{
		return context->ProgramUniform3uiv(program, location, count, value);
	}

	void glProgramUniform4d(GLuint program, GLint location, GLdouble v0, GLdouble v1, GLdouble v2, GLdouble v3)
	{
		return context->ProgramUniform4d(program, location, v0, v1, v2, v3);
	}

	void glProgramUniform4dv(GLuint program, GLint location, GLsizei count, const GLdouble* value)
	{
		return context->ProgramUniform4dv(program, location, count, value);
	}

	void glProgramUniform4f(GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3)
	{
		return context->ProgramUniform4f(program, location, v0, v1, v2, v3);
	}

	void glProgramUniform4fv(GLuint program, GLint location, GLsizei count, const GLfloat* value)
	{
		return context->ProgramUniform4fv(program, location, count, value);
	}

	void glProgramUniform4i(GLuint program, GLint location, GLint v0, GLint v1, GLint v2, GLint v3)
	{
		return context->ProgramUniform4i(program, location, v0, v1, v2, v3);
	}

	void glProgramUniform4iv(GLuint program, GLint location, GLsizei count, const GLint* value)
	{
		return context->ProgramUniform4iv(program, location, count, value);
	}

	void glProgramUniform4ui(GLuint program, GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3)
	{
		return context->ProgramUniform4ui(program, location, v0, v1, v2, v3);
	}

	void glProgramUniform4uiv(GLuint program, GLint location, GLsizei count, const GLuint* value)
	{
		return context->ProgramUniform4uiv(program, location, count, value);
	}

	void glProgramUniformMatrix2dv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->ProgramUniformMatrix2dv(program, location, count, transpose, value);
	}

	void glProgramUniformMatrix2fv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->ProgramUniformMatrix2fv(program, location, count, transpose, value);
	}

	void glProgramUniformMatrix2x3dv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->ProgramUniformMatrix2x3dv(program, location, count, transpose, value);
	}

	void glProgramUniformMatrix2x3fv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->ProgramUniformMatrix2x3fv(program, location, count, transpose, value);
	}

	void glProgramUniformMatrix2x4dv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->ProgramUniformMatrix2x4dv(program, location, count, transpose, value);
	}

	void glProgramUniformMatrix2x4fv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->ProgramUniformMatrix2x4fv(program, location, count, transpose, value);
	}

	void glProgramUniformMatrix3dv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->ProgramUniformMatrix3dv(program, location, count, transpose, value);
	}

	void glProgramUniformMatrix3fv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->ProgramUniformMatrix3fv(program, location, count, transpose, value);
	}

	void glProgramUniformMatrix3x2dv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->ProgramUniformMatrix3x2dv(program, location, count, transpose, value);
	}

	void glProgramUniformMatrix3x2fv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->ProgramUniformMatrix3x2fv(program, location, count, transpose, value);
	}

	void glProgramUniformMatrix3x4dv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->ProgramUniformMatrix3x4dv(program, location, count, transpose, value);
	}

	void glProgramUniformMatrix3x4fv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->ProgramUniformMatrix3x4fv(program, location, count, transpose, value);
	}

	void glProgramUniformMatrix4dv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->ProgramUniformMatrix4dv(program, location, count, transpose, value);
	}

	void glProgramUniformMatrix4fv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->ProgramUniformMatrix4fv(program, location, count, transpose, value);
	}

	void glProgramUniformMatrix4x2dv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->ProgramUniformMatrix4x2dv(program, location, count, transpose, value);
	}

	void glProgramUniformMatrix4x2fv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->ProgramUniformMatrix4x2fv(program, location, count, transpose, value);
	}

	void glProgramUniformMatrix4x3dv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->ProgramUniformMatrix4x3dv(program, location, count, transpose, value);
	}

	void glProgramUniformMatrix4x3fv(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->ProgramUniformMatrix4x3fv(program, location, count, transpose, value);
	}

	void glProvokingVertex(GLenum mode)
	{
		return context->ProvokingVertex(mode);
	}

	void glPushDebugGroup(GLenum source, GLuint id, GLsizei length, const GLchar* message)
	{
		return context->PushDebugGroup(source, id, length, message);
	}

	void glQueryCounter(GLuint id, GLenum target)
	{
		return context->QueryCounter(id, target);
	}

	void glReadBuffer(GLenum src)
	{
		return context->ReadBuffer(src);
	}

	void glReadPixels(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, void* pixels)
	{
		return context->ReadPixels(x, y, width, height, format, type, pixels);
	}

	void glReadnPixels(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void* data)
	{
		return context->ReadnPixels(x, y, width, height, format, type, bufSize, data);
	}

	void glReleaseShaderCompiler()
	{
		return context->ReleaseShaderCompiler();
	}

	void glRenderbufferStorage(GLenum target, GLenum internalformat, GLsizei width, GLsizei height)
	{
		return context->RenderbufferStorage(target, internalformat, width, height);
	}

	void glRenderbufferStorageMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height)
	{
		return context->RenderbufferStorageMultisample(target, samples, internalformat, width, height);
	}

	void glResumeTransformFeedback()
	{
		return context->ResumeTransformFeedback();
	}

	void glSampleCoverage(GLfloat value, GLboolean invert)
	{
		return context->SampleCoverage(value, invert);
	}

	void glSampleMaski(GLuint maskNumber, GLbitfield mask)
	{
		return context->SampleMaski(maskNumber, mask);
	}

	void glSamplerParameterIiv(GLuint sampler, GLenum pname, const GLint* param)
	{
		return context->SamplerParameterIiv(sampler, pname, param);
	}

	void glSamplerParameterIuiv(GLuint sampler, GLenum pname, const GLuint* param)
	{
		return context->SamplerParameterIuiv(sampler, pname, param);
	}

	void glSamplerParameterf(GLuint sampler, GLenum pname, GLfloat param)
	{
		return context->SamplerParameterf(sampler, pname, param);
	}

	void glSamplerParameterfv(GLuint sampler, GLenum pname, const GLfloat* param)
	{
		return context->SamplerParameterfv(sampler, pname, param);
	}

	void glSamplerParameteri(GLuint sampler, GLenum pname, GLint param)
	{
		return context->SamplerParameteri(sampler, pname, param);
	}

	void glSamplerParameteriv(GLuint sampler, GLenum pname, const GLint* param)
	{
		return context->SamplerParameteriv(sampler, pname, param);
	}

	void glScissor(GLint x, GLint y, GLsizei width, GLsizei height)
	{
		return context->Scissor(x, y, width, height);
	}

	void glScissorArrayv(GLuint first, GLsizei count, const GLint* v)
	{
		return context->ScissorArrayv(first, count, v);
	}

	void glScissorIndexed(GLuint index, GLint left, GLint bottom, GLsizei width, GLsizei height)
	{
		return context->ScissorIndexed(index, left, bottom, width, height);
	}

	void glScissorIndexedv(GLuint index, const GLint* v)
	{
		return context->ScissorIndexedv(index, v);
	}

	void glShaderBinary(GLsizei count, const GLuint* shaders, GLenum binaryformat, const void* binary, GLsizei length)
	{
		return context->ShaderBinary(count, shaders, binaryformat, binary, length);
	}

	void glShaderSource(GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length)
	{
		return context->ShaderSource(shader, count, string, length);
	}

	void glShaderStorageBlockBinding(GLuint program, GLuint storageBlockIndex, GLuint storageBlockBinding)
	{
		return context->ShaderStorageBlockBinding(program, storageBlockIndex, storageBlockBinding);
	}

	void glStencilFunc(GLenum func, GLint ref, GLuint mask)
	{
		return context->StencilFunc(func, ref, mask);
	}

	void glStencilFuncSeparate(GLenum face, GLenum func, GLint ref, GLuint mask)
	{
		return context->StencilFuncSeparate(face, func, ref, mask);
	}

	void glStencilMask(GLuint mask)
	{
		return context->StencilMask(mask);
	}

	void glStencilMaskSeparate(GLenum face, GLuint mask)
	{
		return context->StencilMaskSeparate(face, mask);
	}

	void glStencilOp(GLenum fail, GLenum zfail, GLenum zpass)
	{
		return context->StencilOp(fail, zfail, zpass);
	}

	void glStencilOpSeparate(GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass)
	{
		return context->StencilOpSeparate(face, sfail, dpfail, dppass);
	}

	void glTexBuffer(GLenum target, GLenum internalformat, GLuint buffer)
	{
		return context->TexBuffer(target, internalformat, buffer);
	}

	void glTexBufferRange(GLenum target, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size)
	{
		return context->TexBufferRange(target, internalformat, buffer, offset, size);
	}

	void glTexImage1D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border, GLenum format, GLenum type, const void* pixels)
	{
		return context->TexImage1D(target, level, internalformat, width, border, format, type, pixels);
	}

	void glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void* pixels)
	{
		return context->TexImage2D(target, level, internalformat, width, height, border, format, type, pixels);
	}

	void glTexImage2DMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
	{
		return context->TexImage2DMultisample(target, samples, internalformat, width, height, fixedsamplelocations);
	}

	void glTexImage3D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void* pixels)
	{
		return context->TexImage3D(target, level, internalformat, width, height, depth, border, format, type, pixels);
	}

	void glTexImage3DMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
	{
		return context->TexImage3DMultisample(target, samples, internalformat, width, height, depth, fixedsamplelocations);
	}

	void glTexParameterIiv(GLenum target, GLenum pname, const GLint* params)
	{
		return context->TexParameterIiv(target, pname, params);
	}

	void glTexParameterIuiv(GLenum target, GLenum pname, const GLuint* params)
	{
		return context->TexParameterIuiv(target, pname, params);
	}

	void glTexParameterf(GLenum target, GLenum pname, GLfloat param)
	{
		return context->TexParameterf(target, pname, param);
	}

	void glTexParameterfv(GLenum target, GLenum pname, const GLfloat* params)
	{
		return context->TexParameterfv(target, pname, params);
	}

	void glTexParameteri(GLenum target, GLenum pname, GLint param)
	{
		return context->TexParameteri(target, pname, param);
	}

	void glTexParameteriv(GLenum target, GLenum pname, const GLint* params)
	{
		return context->TexParameteriv(target, pname, params);
	}

	void glTexStorage1D(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width)
	{
		return context->TexStorage1D(target, levels, internalformat, width);
	}

	void glTexStorage2D(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height)
	{
		return context->TexStorage2D(target, levels, internalformat, width, height);
	}

	void glTexStorage2DMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
	{
		return context->TexStorage2DMultisample(target, samples, internalformat, width, height, fixedsamplelocations);
	}

	void glTexStorage3D(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth)
	{
		return context->TexStorage3D(target, levels, internalformat, width, height, depth);
	}

	void glTexStorage3DMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
	{
		return context->TexStorage3DMultisample(target, samples, internalformat, width, height, depth, fixedsamplelocations);
	}

	void glTexSubImage1D(GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void* pixels)
	{
		return context->TexSubImage1D(target, level, xoffset, width, format, type, pixels);
	}

	void glTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void* pixels)
	{
		return context->TexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixels);
	}

	void glTexSubImage3D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* pixels)
	{
		return context->TexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
	}

	void glTextureBarrier()
	{
		return context->TextureBarrier();
	}

	void glTextureBuffer(GLuint texture, GLenum internalformat, GLuint buffer)
	{
		return context->TextureBuffer(texture, internalformat, buffer);
	}

	void glTextureBufferRange(GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size)
	{
		return context->TextureBufferRange(texture, internalformat, buffer, offset, size);
	}

	void glTextureParameterIiv(GLuint texture, GLenum pname, const GLint* params)
	{
		return context->TextureParameterIiv(texture, pname, params);
	}

	void glTextureParameterIuiv(GLuint texture, GLenum pname, const GLuint* params)
	{
		return context->TextureParameterIuiv(texture, pname, params);
	}

	void glTextureParameterf(GLuint texture, GLenum pname, GLfloat param)
	{
		return context->TextureParameterf(texture, pname, param);
	}

	void glTextureParameterfv(GLuint texture, GLenum pname, const GLfloat* param)
	{
		return context->TextureParameterfv(texture, pname, param);
	}

	void glTextureParameteri(GLuint texture, GLenum pname, GLint param)
	{
		return context->TextureParameteri(texture, pname, param);
	}

	void glTextureParameteriv(GLuint texture, GLenum pname, const GLint* param)
	{
		return context->TextureParameteriv(texture, pname, param);
	}

	void glTextureStorage1D(GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width)
	{
		return context->TextureStorage1D(texture, levels, internalformat, width);
	}

	void glTextureStorage2D(GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height)
	{
		return context->TextureStorage2D(texture, levels, internalformat, width, height);
	}

	void glTextureStorage2DMultisample(GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
	{
		return context->TextureStorage2DMultisample(texture, samples, internalformat, width, height, fixedsamplelocations);
	}

	void glTextureStorage3D(GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth)
	{
		return context->TextureStorage3D(texture, levels, internalformat, width, height, depth);
	}

	void glTextureStorage3DMultisample(GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
	{
		return context->TextureStorage3DMultisample(texture, samples, internalformat, width, height, depth, fixedsamplelocations);
	}

	void glTextureSubImage1D(GLuint texture, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void* pixels)
	{
		return context->TextureSubImage1D(texture, level, xoffset, width, format, type, pixels);
	}

	void glTextureSubImage2D(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void* pixels)
	{
		return context->TextureSubImage2D(texture, level, xoffset, yoffset, width, height, format, type, pixels);
	}

	void glTextureSubImage3D(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* pixels)
	{
		return context->TextureSubImage3D(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
	}

	void glTextureView(GLuint texture, GLenum target, GLuint origtexture, GLenum internalformat, GLuint minlevel, GLuint numlevels, GLuint minlayer, GLuint numlayers)
	{
		return context->TextureView(texture, target, origtexture, internalformat, minlevel, numlevels, minlayer, numlayers);
	}

	void glTransformFeedbackBufferBase(GLuint xfb, GLuint index, GLuint buffer)
	{
		return context->TransformFeedbackBufferBase(xfb, index, buffer);
	}

	void glTransformFeedbackBufferRange(GLuint xfb, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size)
	{
		return context->TransformFeedbackBufferRange(xfb, index, buffer, offset, size);
	}

	void glTransformFeedbackVaryings(GLuint program, GLsizei count, const GLchar* const* varyings, GLenum bufferMode)
	{
		return context->TransformFeedbackVaryings(program, count, varyings, bufferMode);
	}

	void glUniform1d(GLint location, GLdouble x)
	{
		return context->Uniform1d(location, x);
	}

	void glUniform1dv(GLint location, GLsizei count, const GLdouble* value)
	{
		return context->Uniform1dv(location, count, value);
	}

	void glUniform1f(GLint location, GLfloat v0)
	{
		return context->Uniform1f(location, v0);
	}

	void glUniform1fv(GLint location, GLsizei count, const GLfloat* value)
	{
		return context->Uniform1fv(location, count, value);
	}

	void glUniform1i(GLint location, GLint v0)
	{
		return context->Uniform1i(location, v0);
	}

	void glUniform1iv(GLint location, GLsizei count, const GLint* value)
	{
		return context->Uniform1iv(location, count, value);
	}

	void glUniform1ui(GLint location, GLuint v0)
	{
		return context->Uniform1ui(location, v0);
	}

	void glUniform1uiv(GLint location, GLsizei count, const GLuint* value)
	{
		return context->Uniform1uiv(location, count, value);
	}

	void glUniform2d(GLint location, GLdouble x, GLdouble y)
	{
		return context->Uniform2d(location, x, y);
	}

	void glUniform2dv(GLint location, GLsizei count, const GLdouble* value)
	{
		return context->Uniform2dv(location, count, value);
	}

	void glUniform2f(GLint location, GLfloat v0, GLfloat v1)
	{
		return context->Uniform2f(location, v0, v1);
	}

	void glUniform2fv(GLint location, GLsizei count, const GLfloat* value)
	{
		return context->Uniform2fv(location, count, value);
	}

	void glUniform2i(GLint location, GLint v0, GLint v1)
	{
		return context->Uniform2i(location, v0, v1);
	}

	void glUniform2iv(GLint location, GLsizei count, const GLint* value)
	{
		return context->Uniform2iv(location, count, value);
	}

	void glUniform2ui(GLint location, GLuint v0, GLuint v1)
	{
		return context->Uniform2ui(location, v0, v1);
	}

	void glUniform2uiv(GLint location, GLsizei count, const GLuint* value)
	{
		return context->Uniform2uiv(location, count, value);
	}

	void glUniform3d(GLint location, GLdouble x, GLdouble y, GLdouble z)
	{
		return context->Uniform3d(location, x, y, z);
	}

	void glUniform3dv(GLint location, GLsizei count, const GLdouble* value)
	{
		return context->Uniform3dv(location, count, value);
	}

	void glUniform3f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2)
	{
		return context->Uniform3f(location, v0, v1, v2);
	}

	void glUniform3fv(GLint location, GLsizei count, const GLfloat* value)
	{
		return context->Uniform3fv(location, count, value);
	}

	void glUniform3i(GLint location, GLint v0, GLint v1, GLint v2)
	{
		return context->Uniform3i(location, v0, v1, v2);
	}

	void glUniform3iv(GLint location, GLsizei count, const GLint* value)
	{
		return context->Uniform3iv(location, count, value);
	}

	void glUniform3ui(GLint location, GLuint v0, GLuint v1, GLuint v2)
	{
		return context->Uniform3ui(location, v0, v1, v2);
	}

	void glUniform3uiv(GLint location, GLsizei count, const GLuint* value)
	{
		return context->Uniform3uiv(location, count, value);
	}

	void glUniform4d(GLint location, GLdouble x, GLdouble y, GLdouble z, GLdouble w)
	{
		return context->Uniform4d(location, x, y, z, w);
	}

	void glUniform4dv(GLint location, GLsizei count, const GLdouble* value)
	{
		return context->Uniform4dv(location, count, value);
	}

	void glUniform4f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3)
	{
		return context->Uniform4f(location, v0, v1, v2, v3);
	}

	void glUniform4fv(GLint location, GLsizei count, const GLfloat* value)
	{
		return context->Uniform4fv(location, count, value);
	}

	void glUniform4i(GLint location, GLint v0, GLint v1, GLint v2, GLint v3)
	{
		return context->Uniform4i(location, v0, v1, v2, v3);
	}

	void glUniform4iv(GLint location, GLsizei count, const GLint* value)
	{
		return context->Uniform4iv(location, count, value);
	}

	void glUniform4ui(GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3)
	{
		return context->Uniform4ui(location, v0, v1, v2, v3);
	}

	void glUniform4uiv(GLint location, GLsizei count, const GLuint* value)
	{
		return context->Uniform4uiv(location, count, value);
	}

	void glUniformBlockBinding(GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding)
	{
		return context->UniformBlockBinding(program, uniformBlockIndex, uniformBlockBinding);
	}

	void glUniformMatrix2dv(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->UniformMatrix2dv(location, count, transpose, value);
	}

	void glUniformMatrix2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->UniformMatrix2fv(location, count, transpose, value);
	}

	void glUniformMatrix2x3dv(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->UniformMatrix2x3dv(location, count, transpose, value);
	}

	void glUniformMatrix2x3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->UniformMatrix2x3fv(location, count, transpose, value);
	}

	void glUniformMatrix2x4dv(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->UniformMatrix2x4dv(location, count, transpose, value);
	}

	void glUniformMatrix2x4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->UniformMatrix2x4fv(location, count, transpose, value);
	}

	void glUniformMatrix3dv(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->UniformMatrix3dv(location, count, transpose, value);
	}

	void glUniformMatrix3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->UniformMatrix3fv(location, count, transpose, value);
	}

	void glUniformMatrix3x2dv(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->UniformMatrix3x2dv(location, count, transpose, value);
	}

	void glUniformMatrix3x2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->UniformMatrix3x2fv(location, count, transpose, value);
	}

	void glUniformMatrix3x4dv(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->UniformMatrix3x4dv(location, count, transpose, value);
	}

	void glUniformMatrix3x4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->UniformMatrix3x4fv(location, count, transpose, value);
	}

	void glUniformMatrix4dv(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->UniformMatrix4dv(location, count, transpose, value);
	}

	void glUniformMatrix4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->UniformMatrix4fv(location, count, transpose, value);
	}

	void glUniformMatrix4x2dv(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->UniformMatrix4x2dv(location, count, transpose, value);
	}

	void glUniformMatrix4x2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->UniformMatrix4x2fv(location, count, transpose, value);
	}

	void glUniformMatrix4x3dv(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value)
	{
		return context->UniformMatrix4x3dv(location, count, transpose, value);
	}

	void glUniformMatrix4x3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)
	{
		return context->UniformMatrix4x3fv(location, count, transpose, value);
	}

	void glUniformSubroutinesuiv(GLenum shadertype, GLsizei count, const GLuint* indices)
	{
		return context->UniformSubroutinesuiv(shadertype, count, indices);
	}

	GLboolean glUnmapBuffer(GLenum target)
	{
		return context->UnmapBuffer(target);
	}

	GLboolean glUnmapNamedBuffer(GLuint buffer)
	{
		return context->UnmapNamedBuffer(buffer);
	}

	void glUseProgram(GLuint program)
	{
		return context->UseProgram(program);
	}

	void glUseProgramStages(GLuint pipeline, GLbitfield stages, GLuint program)
	{
		return context->UseProgramStages(pipeline, stages, program);
	}

	void glValidateProgram(GLuint program)
	{
		return context->ValidateProgram(program);
	}

	void glValidateProgramPipeline(GLuint pipeline)
	{
		return context->ValidateProgramPipeline(pipeline);
	}

	void glVertexArrayAttribBinding(GLuint vaobj, GLuint attribindex, GLuint bindingindex)
	{
		return context->VertexArrayAttribBinding(vaobj, attribindex, bindingindex);
	}

	void glVertexArrayAttribFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset)
	{
		return context->VertexArrayAttribFormat(vaobj, attribindex, size, type, normalized, relativeoffset);
	}

	void glVertexArrayAttribIFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
	{
		return context->VertexArrayAttribIFormat(vaobj, attribindex, size, type, relativeoffset);
	}

	void glVertexArrayAttribLFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
	{
		return context->VertexArrayAttribLFormat(vaobj, attribindex, size, type, relativeoffset);
	}

	void glVertexArrayBindingDivisor(GLuint vaobj, GLuint bindingindex, GLuint divisor)
	{
		return context->VertexArrayBindingDivisor(vaobj, bindingindex, divisor);
	}

	void glVertexArrayElementBuffer(GLuint vaobj, GLuint buffer)
	{
		return context->VertexArrayElementBuffer(vaobj, buffer);
	}

	void glVertexArrayVertexBuffer(GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride)
	{
		return context->VertexArrayVertexBuffer(vaobj, bindingindex, buffer, offset, stride);
	}

	void glVertexArrayVertexBuffers(GLuint vaobj, GLuint first, GLsizei count, const GLuint* buffers, const GLintptr* offsets, const GLsizei* strides)
	{
		return context->VertexArrayVertexBuffers(vaobj, first, count, buffers, offsets, strides);
	}

	void glVertexAttrib1d(GLuint index, GLdouble x)
	{
		return context->VertexAttrib1d(index, x);
	}

	void glVertexAttrib1dv(GLuint index, const GLdouble* v)
	{
		return context->VertexAttrib1dv(index, v);
	}

	void glVertexAttrib1f(GLuint index, GLfloat x)
	{
		return context->VertexAttrib1f(index, x);
	}

	void glVertexAttrib1fv(GLuint index, const GLfloat* v)
	{
		return context->VertexAttrib1fv(index, v);
	}

	void glVertexAttrib1s(GLuint index, GLshort x)
	{
		return context->VertexAttrib1s(index, x);
	}

	void glVertexAttrib1sv(GLuint index, const GLshort* v)
	{
		return context->VertexAttrib1sv(index, v);
	}

	void glVertexAttrib2d(GLuint index, GLdouble x, GLdouble y)
	{
		return context->VertexAttrib2d(index, x, y);
	}

	void glVertexAttrib2dv(GLuint index, const GLdouble* v)
	{
		return context->VertexAttrib2dv(index, v);
	}

	void glVertexAttrib2f(GLuint index, GLfloat x, GLfloat y)
	{
		return context->VertexAttrib2f(index, x, y);
	}

	void glVertexAttrib2fv(GLuint index, const GLfloat* v)
	{
		return context->VertexAttrib2fv(index, v);
	}

	void glVertexAttrib2s(GLuint index, GLshort x, GLshort y)
	{
		return context->VertexAttrib2s(index, x, y);
	}

	void glVertexAttrib2sv(GLuint index, const GLshort* v)
	{
		return context->VertexAttrib2sv(index, v);
	}

	void glVertexAttrib3d(GLuint index, GLdouble x, GLdouble y, GLdouble z)
	{
		return context->VertexAttrib3d(index, x, y, z);
	}

	void glVertexAttrib3dv(GLuint index, const GLdouble* v)
	{
		return context->VertexAttrib3dv(index, v);
	}

	void glVertexAttrib3f(GLuint index, GLfloat x, GLfloat y, GLfloat z)
	{
		return context->VertexAttrib3f(index, x, y, z);
	}

	void glVertexAttrib3fv(GLuint index, const GLfloat* v)
	{
		return context->VertexAttrib3fv(index, v);
	}

	void glVertexAttrib3s(GLuint index, GLshort x, GLshort y, GLshort z)
	{
		return context->VertexAttrib3s(index, x, y, z);
	}

	void glVertexAttrib3sv(GLuint index, const GLshort* v)
	{
		return context->VertexAttrib3sv(index, v);
	}

	void glVertexAttrib4Nbv(GLuint index, const GLbyte* v)
	{
		return context->VertexAttrib4Nbv(index, v);
	}

	void glVertexAttrib4Niv(GLuint index, const GLint* v)
	{
		return context->VertexAttrib4Niv(index, v);
	}

	void glVertexAttrib4Nsv(GLuint index, const GLshort* v)
	{
		return context->VertexAttrib4Nsv(index, v);
	}

	void glVertexAttrib4Nub(GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w)
	{
		return context->VertexAttrib4Nub(index, x, y, z, w);
	}

	void glVertexAttrib4Nubv(GLuint index, const GLubyte* v)
	{
		return context->VertexAttrib4Nubv(index, v);
	}

	void glVertexAttrib4Nuiv(GLuint index, const GLuint* v)
	{
		return context->VertexAttrib4Nuiv(index, v);
	}

	void glVertexAttrib4Nusv(GLuint index, const GLushort* v)
	{
		return context->VertexAttrib4Nusv(index, v);
	}

	void glVertexAttrib4bv(GLuint index, const GLbyte* v)
	{
		return context->VertexAttrib4bv(index, v);
	}

	void glVertexAttrib4d(GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w)
	{
		return context->VertexAttrib4d(index, x, y, z, w);
	}

	void glVertexAttrib4dv(GLuint index, const GLdouble* v)
	{
		return context->VertexAttrib4dv(index, v);
	}

	void glVertexAttrib4f(GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
	{
		return context->VertexAttrib4f(index, x, y, z, w);
	}

	void glVertexAttrib4fv(GLuint index, const GLfloat* v)
	{
		return context->VertexAttrib4fv(index, v);
	}

	void glVertexAttrib4iv(GLuint index, const GLint* v)
	{
		return context->VertexAttrib4iv(index, v);
	}

	void glVertexAttrib4s(GLuint index, GLshort x, GLshort y, GLshort z, GLshort w)
	{
		return context->VertexAttrib4s(index, x, y, z, w);
	}

	void glVertexAttrib4sv(GLuint index, const GLshort* v)
	{
		return context->VertexAttrib4sv(index, v);
	}

	void glVertexAttrib4ubv(GLuint index, const GLubyte* v)
	{
		return context->VertexAttrib4ubv(index, v);
	}

	void glVertexAttrib4uiv(GLuint index, const GLuint* v)
	{
		return context->VertexAttrib4uiv(index, v);
	}

	void glVertexAttrib4usv(GLuint index, const GLushort* v)
	{
		return context->VertexAttrib4usv(index, v);
	}

	void glVertexAttribBinding(GLuint attribindex, GLuint bindingindex)
	{
		return context->VertexAttribBinding(attribindex, bindingindex);
	}

	void glVertexAttribDivisor(GLuint index, GLuint divisor)
	{
		return context->VertexAttribDivisor(index, divisor);
	}

	void glVertexAttribFormat(GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset)
	{
		return context->VertexAttribFormat(attribindex, size, type, normalized, relativeoffset);
	}

	void glVertexAttribI1i(GLuint index, GLint x)
	{
		return context->VertexAttribI1i(index, x);
	}

	void glVertexAttribI1iv(GLuint index, const GLint* v)
	{
		return context->VertexAttribI1iv(index, v);
	}

	void glVertexAttribI1ui(GLuint index, GLuint x)
	{
		return context->VertexAttribI1ui(index, x);
	}

	void glVertexAttribI1uiv(GLuint index, const GLuint* v)
	{
		return context->VertexAttribI1uiv(index, v);
	}

	void glVertexAttribI2i(GLuint index, GLint x, GLint y)
	{
		return context->VertexAttribI2i(index, x, y);
	}

	void glVertexAttribI2iv(GLuint index, const GLint* v)
	{
		return context->VertexAttribI2iv(index, v);
	}

	void glVertexAttribI2ui(GLuint index, GLuint x, GLuint y)
	{
		return context->VertexAttribI2ui(index, x, y);
	}

	void glVertexAttribI2uiv(GLuint index, const GLuint* v)
	{
		return context->VertexAttribI2uiv(index, v);
	}

	void glVertexAttribI3i(GLuint index, GLint x, GLint y, GLint z)
	{
		return context->VertexAttribI3i(index, x, y, z);
	}

	void glVertexAttribI3iv(GLuint index, const GLint* v)
	{
		return context->VertexAttribI3iv(index, v);
	}

	void glVertexAttribI3ui(GLuint index, GLuint x, GLuint y, GLuint z)
	{
		return context->VertexAttribI3ui(index, x, y, z);
	}

	void glVertexAttribI3uiv(GLuint index, const GLuint* v)
	{
		return context->VertexAttribI3uiv(index, v);
	}

	void glVertexAttribI4bv(GLuint index, const GLbyte* v)
	{
		return context->VertexAttribI4bv(index, v);
	}

	void glVertexAttribI4i(GLuint index, GLint x, GLint y, GLint z, GLint w)
	{
		return context->VertexAttribI4i(index, x, y, z, w);
	}

	void glVertexAttribI4iv(GLuint index, const GLint* v)
	{
		return context->VertexAttribI4iv(index, v);
	}

	void glVertexAttribI4sv(GLuint index, const GLshort* v)
	{
		return context->VertexAttribI4sv(index, v);
	}

	void glVertexAttribI4ubv(GLuint index, const GLubyte* v)
	{
		return context->VertexAttribI4ubv(index, v);
	}

	void glVertexAttribI4ui(GLuint index, GLuint x, GLuint y, GLuint z, GLuint w)
	{
		return context->VertexAttribI4ui(index, x, y, z, w);
	}

	void glVertexAttribI4uiv(GLuint index, const GLuint* v)
	{
		return context->VertexAttribI4uiv(index, v);
	}

	void glVertexAttribI4usv(GLuint index, const GLushort* v)
	{
		return context->VertexAttribI4usv(index, v);
	}

	void glVertexAttribIFormat(GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
	{
		return context->VertexAttribIFormat(attribindex, size, type, relativeoffset);
	}

	void glVertexAttribIPointer(GLuint index, GLint size, GLenum type, GLsizei stride, const void* pointer)
	{
		return context->VertexAttribIPointer(index, size, type, stride, pointer);
	}

	void glVertexAttribL1d(GLuint index, GLdouble x)
	{
		return context->VertexAttribL1d(index, x);
	}

	void glVertexAttribL1dv(GLuint index, const GLdouble* v)
	{
		return context->VertexAttribL1dv(index, v);
	}

	void glVertexAttribL2d(GLuint index, GLdouble x, GLdouble y)
	{
		return context->VertexAttribL2d(index, x, y);
	}

	void glVertexAttribL2dv(GLuint index, const GLdouble* v)
	{
		return context->VertexAttribL2dv(index, v);
	}

	void glVertexAttribL3d(GLuint index, GLdouble x, GLdouble y, GLdouble z)
	{
		return context->VertexAttribL3d(index, x, y, z);
	}

	void glVertexAttribL3dv(GLuint index, const GLdouble* v)
	{
		return context->VertexAttribL3dv(index, v);
	}

	void glVertexAttribL4d(GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w)
	{
		return context->VertexAttribL4d(index, x, y, z, w);
	}

	void glVertexAttribL4dv(GLuint index, const GLdouble* v)
	{
		return context->VertexAttribL4dv(index, v);
	}

	void glVertexAttribLFormat(GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
	{
		return context->VertexAttribLFormat(attribindex, size, type, relativeoffset);
	}

	void glVertexAttribLPointer(GLuint index, GLint size, GLenum type, GLsizei stride, const void* pointer)
	{
		return context->VertexAttribLPointer(index, size, type, stride, pointer);
	}

	void glVertexAttribP1ui(GLuint index, GLenum type, GLboolean normalized, GLuint value)
	{
		return context->VertexAttribP1ui(index, type, normalized, value);
	}

	void glVertexAttribP1uiv(GLuint index, GLenum type, GLboolean normalized, const GLuint* value)
	{
		return context->VertexAttribP1uiv(index, type, normalized, value);
	}

	void glVertexAttribP2ui(GLuint index, GLenum type, GLboolean normalized, GLuint value)
	{
		return context->VertexAttribP2ui(index, type, normalized, value);
	}

	void glVertexAttribP2uiv(GLuint index, GLenum type, GLboolean normalized, const GLuint* value)
	{
		return context->VertexAttribP2uiv(index, type, normalized, value);
	}

	void glVertexAttribP3ui(GLuint index, GLenum type, GLboolean normalized, GLuint value)
	{
		return context->VertexAttribP3ui(index, type, normalized, value);
	}

	void glVertexAttribP3uiv(GLuint index, GLenum type, GLboolean normalized, const GLuint* value)
	{
		return context->VertexAttribP3uiv(index, type, normalized, value);
	}

	void glVertexAttribP4ui(GLuint index, GLenum type, GLboolean normalized, GLuint value)
	{
		return context->VertexAttribP4ui(index, type, normalized, value);
	}

	void glVertexAttribP4uiv(GLuint index, GLenum type, GLboolean normalized, const GLuint* value)
	{
		return context->VertexAttribP4uiv(index, type, normalized, value);
	}

	void glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer)
	{
		return context->VertexAttribPointer(index, size, type, normalized, stride, pointer);
	}

	void glVertexBindingDivisor(GLuint bindingindex, GLuint divisor)
	{
		return context->VertexBindingDivisor(bindingindex, divisor);
	}

	void glViewport(GLint x, GLint y, GLsizei width, GLsizei height)
	{
		return context->Viewport(x, y, width, height);
	}

	void glViewportArrayv(GLuint first, GLsizei count, const GLfloat* v)
	{
		return context->ViewportArrayv(first, count, v);
	}

	void glViewportIndexedf(GLuint index, GLfloat x, GLfloat y, GLfloat w, GLfloat h)
	{
		return context->ViewportIndexedf(index, x, y, w, h);
	}

	void glViewportIndexedfv(GLuint index, const GLfloat* v)
	{
		return context->ViewportIndexedfv(index, v);
	}

	void glWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout)
	{
		return context->WaitSync(sync, flags, timeout);
	}
}
