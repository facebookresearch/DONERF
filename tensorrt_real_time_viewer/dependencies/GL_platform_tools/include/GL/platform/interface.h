


#ifndef INCLUDED_GL_PLATFORM_INTERFACE
#define INCLUDED_GL_PLATFORM_INTERFACE

#pragma once


#ifdef _MSC_VER
#define INTERFACE __declspec(novtable)
#else
#define INTERFACE
#endif


#endif  // INCLUDED_GL_PLATFORM_INTERFACE
