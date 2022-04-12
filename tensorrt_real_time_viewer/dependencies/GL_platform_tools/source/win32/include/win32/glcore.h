


#ifndef INCLUDED_WIN32_GLCORE
#define INCLUDED_WIN32_GLCORE

#pragma once


#ifndef GLCOREAPI
#ifdef GLCORE_STATIC
#define GLCOREAPI
#else
#define GLCOREAPI __declspec(dllimport)
#endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif

struct glcoreContext;

GLCOREAPI const glcoreContext* APIENTRY glcoreContextInit();
GLCOREAPI void APIENTRY glcoreContextDestroy(const glcoreContext* ctx);
GLCOREAPI void APIENTRY glcoreContextMakeCurrent(const glcoreContext* ctx);
GLCOREAPI const glcoreContext* APIENTRY glcoreContextGetCurrent();

#ifdef __cplusplus
}
#endif

#endif  // INCLUDED_WIN32_GLCORE
