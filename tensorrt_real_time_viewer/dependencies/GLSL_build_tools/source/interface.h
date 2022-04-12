


#ifndef INCLUDED_INTERFACE_H
#define INCLUDED_INTERFACE_H

#pragma once


#ifdef _MSC_VER
#define INTERFACE __declspec(novtable)
#else
#define INTERFACE
#endif


#endif  // INCLUDED_INTERFACE_H
