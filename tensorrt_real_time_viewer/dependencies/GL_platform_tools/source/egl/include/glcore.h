


#ifndef INCLUDED_EGL_GLCORE
#define INCLUDED_EGL_GLCORE

#pragma once


#ifdef __cplusplus
extern "C"
{
#endif

struct glcoreContext;

const glcoreContext* APIENTRY glcoreContextInit();
void APIENTRY glcoreContextDestroy(const glcoreContext* ctx);
void APIENTRY glcoreContextMakeCurrent(const glcoreContext* ctx);
const glcoreContext* APIENTRY glcoreContextGetCurrent();

#ifdef __cplusplus
}
#endif

#endif  // INCLUDED_EGL_GLCORE
