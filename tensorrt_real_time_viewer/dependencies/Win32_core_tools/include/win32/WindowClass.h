


#ifndef INCLUDED_WIN32_WINDOW_CLASS
#define INCLUDED_WIN32_WINDOW_CLASS

#pragma once

#include <win32/platform.h>

#include "window_handle.h"


namespace Win32
{
	template <class T, LRESULT(T::*WndProc)(HWND, UINT, WPARAM, LPARAM)>
	class WindowClass
	{
	private:
		ATOM cls;

		static LRESULT CALLBACK BootstrapWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
		{
			if (msg == WM_CREATE)
			{
				T* obj = static_cast<T*>(reinterpret_cast<CREATESTRUCT*>(lParam)->lpCreateParams);
				SetWindowLongPtrW(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(obj));
				SetWindowLongPtrW(hWnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&WndProcThunk));
				return (obj->*WndProc)(hWnd, msg, wParam, lParam);
			}

			return DefWindowProcW(hWnd, msg, wParam, lParam);
		}

		static LRESULT CALLBACK WndProcThunk(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
		{
			T* obj = reinterpret_cast<T*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));
			return (obj->*WndProc)(hWnd, msg, wParam, lParam);
		}

	public:
		WindowClass(LPCWSTR lpszClassName,
		            UINT style,
		            HICON hIcon,
		            HICON hIconSm,
		            HCURSOR hCursor,
		            HBRUSH hbrBackground = (HBRUSH) (COLOR_WINDOW + 1),
		            LPCWSTR lpszMenuName = nullptr)
		{
			WNDCLASSEXW wnd_cls;
			wnd_cls.cbSize = sizeof(wnd_cls);
			wnd_cls.style = style;
			wnd_cls.lpfnWndProc = &BootstrapWndProc;
			wnd_cls.cbClsExtra = 0;
			wnd_cls.cbWndExtra = 0;
			wnd_cls.hInstance = GetModuleHandleW(nullptr);
			wnd_cls.hIcon = hIcon;
			wnd_cls.hCursor = hCursor;
			wnd_cls.hbrBackground = hbrBackground;
			wnd_cls.lpszMenuName = lpszMenuName;
			wnd_cls.lpszClassName = lpszClassName;
			wnd_cls.hIconSm = hIconSm;
			cls = RegisterClassExW(&wnd_cls);
		}

		~WindowClass()
		{
			UnregisterClassW(reinterpret_cast<LPCWSTR>(cls), GetModuleHandleW(nullptr));
		}

		unique_hwnd createWindow(T& obj,
		                         DWORD dwExStyle,
		                         LPCWSTR lpWindowName,
		                         DWORD dwStyle,
		                         int X = CW_USEDEFAULT,
		                         int Y = CW_USEDEFAULT,
		                         int nWidth = CW_USEDEFAULT,
		                         int nHeight = CW_USEDEFAULT,
		                         HWND hWndParent = 0,
		                         HMENU hMenu = 0) const
		{
			return unique_hwnd(
				CreateWindowExW(dwExStyle,
				                reinterpret_cast<LPCWSTR>(cls),
				                lpWindowName,
				                dwStyle,
				                X,
				                Y,
				                nWidth,
				                nHeight,
				                hWndParent,
				                hMenu,
				                GetModuleHandleW(nullptr),
				                &obj));
		}
	};
}

#endif  // INCLUDED_WIN32_WINDOW_CLASS
