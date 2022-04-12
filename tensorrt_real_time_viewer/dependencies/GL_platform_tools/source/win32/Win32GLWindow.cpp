


#include <limits>
#include <cassert>
#include <stdexcept>

#include <string>

#include <win32/window_handle.h>
#include <win32/WindowClass.h>
#include <win32/unicode.h>

#include "Win32GLApplication.h"
#include "Win32GLWindow.h"


namespace Win32
{
	namespace GL
	{
		void setPixelFormat(HWND hwnd, int depth_buffer_bits, int stencil_buffer_bits, bool stereo)
		{
			auto hdc_deleter = [hwnd](HDC hdc)
			{
				ReleaseDC(hwnd, hdc);
			};

			unique_handle<HDC, 0, decltype(hdc_deleter)> hdc(GetDC(hwnd), hdc_deleter);
			setPixelFormat(hdc, depth_buffer_bits, stencil_buffer_bits, stereo);
		}

		Context createContext(HWND hwnd, int version_major, int version_minor, bool debug)
		{
			auto hdc_deleter = [hwnd](HDC hdc)
			{
				ReleaseDC(hwnd, hdc);
			};

			unique_handle<HDC, 0, decltype(hdc_deleter)> hdc(GetDC(hwnd), hdc_deleter);
			return Context(hdc, version_major, version_minor, debug);
		}


		unique_hwnd Window::createWindow(Window& wnd, DWORD dwExStyle, LPCWSTR lpWindowName, DWORD dwStyle, int X, int Y, int nWidth, int nHeight, HWND hWndParent, HMENU hMenu)
		{
			static Win32::WindowClass<Window, &Window::WindowProc> wnd_cls(L"Win32GLWindow", 0, LoadIcon(0, IDI_APPLICATION), 0, LoadCursor(0, IDC_ARROW), 0, 0);
			Win32::unique_hwnd hwnd = wnd_cls.createWindow(wnd, dwExStyle, lpWindowName, dwStyle, X, Y, nWidth, nHeight, hWndParent, hMenu);

			if (hwnd == 0)
				throw std::runtime_error("CreateWindowEx() failed");

			return hwnd;
		}

		unique_hwnd Window::createWindow(Window& wnd, const char* title, int x, int y, int width, int height)
		{
			RECT r;
			r.left = 0;
			r.top = 0;
			r.right = width;
			r.bottom = height;
			AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
			width = r.right - r.left;
			height = r.bottom - r.top;

			return createWindow(wnd, 0U, widen(title).c_str(), WS_OVERLAPPEDWINDOW, x, y, width, height, 0, 0);
		}

		unique_hwnd Window::createWindow(Window& wnd, const char* title)
		{
			return createWindow(wnd, 0U, widen(title).c_str(), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, 0, 0);
		}


		Window::Window(const char* title, int x, int y, int width, int height, int depth_buffer_bits, int stencil_buffer_bits, bool stereo, bool fullscreen)
			: hwnd(createWindow(*this, title, x, y, width, height)),
			  display_handler(nullptr),
			  mouse_input(nullptr),
			  keyboard_input(nullptr)
		{
			if (fullscreen)
				toggleFullscreen();
			setPixelFormat(hwnd, depth_buffer_bits, stencil_buffer_bits, stereo);
			ShowWindow(hwnd, SW_SHOWNORMAL);
		}

		Window::Window(const char* title, int width, int height, int depth_buffer_bits, int stencil_buffer_bits, bool stereo, bool fullscreen)
			: Window(title, CW_USEDEFAULT, CW_USEDEFAULT, width, height, depth_buffer_bits, stencil_buffer_bits, stereo, fullscreen)
		{
		}

		Window::Window(const char* title, const WINDOWPLACEMENT& placement, int depth_buffer_bits, int stencil_buffer_bits, bool stereo)
			: hwnd(createWindow(*this, title)),
			  display_handler(nullptr),
			  mouse_input(nullptr),
			  keyboard_input(nullptr)
		{
			setPixelFormat(hwnd, depth_buffer_bits, stencil_buffer_bits, stereo);
			ShowWindow(hwnd, SW_SHOWNORMAL);
			place(placement);
		}

		void Window::savePlacement(WINDOWPLACEMENT& placement) const
		{
			placement.length = sizeof(WINDOWPLACEMENT);
			GetWindowPlacement(hwnd, &placement);
		}

		void Window::place(const WINDOWPLACEMENT& placement)
		{
			SetWindowPlacement(hwnd, &placement);
		}

		void Window::title(const char* title)
		{
			SetWindowTextW(hwnd, widen(title).c_str());
		}

		void Window::resize(int width, int height)
		{
			RECT cr;
			cr.left = 0;
			cr.top = 0;
			cr.right = width;
			cr.bottom = height;
			AdjustWindowRect(&cr, WS_OVERLAPPEDWINDOW, FALSE);

			SetWindowPos(hwnd, 0, cr.left, cr.top, cr.right - cr.left, cr.bottom - cr.top, SWP_NOMOVE | SWP_NOACTIVATE | SWP_NOZORDER);
		}

		void Window::toggleFullscreen()
		{
			DWORD style = GetWindowLongW(hwnd, GWL_STYLE);

			if (style & WS_OVERLAPPEDWINDOW)
			{
				if (GetWindowPlacement(hwnd, &windowed_placement))
				{
					MONITORINFOEXW monitor_info;
					monitor_info.cbSize = sizeof(monitor_info);

					GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &monitor_info);
					SetWindowLongW(hwnd, GWL_STYLE, (style & ~WS_OVERLAPPEDWINDOW) | WS_POPUP);
					SetWindowPos(hwnd,
					             HWND_TOP,
					             monitor_info.rcMonitor.left,
					             monitor_info.rcMonitor.top,
					             monitor_info.rcMonitor.right - monitor_info.rcMonitor.left,
					             monitor_info.rcMonitor.bottom - monitor_info.rcMonitor.top,
					             SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
				}
			}
			else
			{
				SetWindowLongW(hwnd, GWL_STYLE, (style & ~WS_POPUP) | WS_OVERLAPPEDWINDOW);
				SetWindowPlacement(hwnd, &windowed_placement);
				SetWindowPos(hwnd, 0, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
			}
		}

		Context Window::createContext(int version_major, int version_minor, bool debug)
		{
			return Win32::GL::createContext(hwnd, version_major, version_minor, debug);
		}

		void Window::attach(::GL::platform::DisplayHandler* handler)
		{
			display_handler = handler;
			if (display_handler)
			{
				RECT rc;
				GetClientRect(hwnd, &rc);
				display_handler->resize(rc.right - rc.left, rc.bottom - rc.top);
				display_handler->move(rc.left, rc.top);
			}
		}

		void Window::attach(::GL::platform::MouseInputHandler* handler)
		{
			mouse_input = handler;
		}

		void Window::attach(::GL::platform::KeyboardInputHandler* handler)
		{
			keyboard_input = handler;
		}

		namespace
		{
			::GL::platform::Key keycode(WPARAM wParam, LPARAM lParam)
			{
				return static_cast<::GL::platform::Key>(wParam);
			}
		}

		LRESULT Window::WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
		{
			switch (msg)
			{
			case WM_CLOSE:
				if (display_handler)
					display_handler->close();
				else
					quit();
				break;

			case WM_DESTROY:
				if (display_handler)
					display_handler->destroy();
				break;

			case WM_LBUTTONDOWN:
				if (mouse_input)
					mouse_input->buttonDown(::GL::platform::Button::LEFT, LOWORD(lParam), HIWORD(lParam));
				break;

			case WM_MBUTTONDOWN:
				if (mouse_input)
					mouse_input->buttonDown(::GL::platform::Button::MIDDLE, LOWORD(lParam), HIWORD(lParam));
				break;

			case WM_RBUTTONDOWN:
				if (mouse_input)
					mouse_input->buttonDown(::GL::platform::Button::RIGHT, LOWORD(lParam), HIWORD(lParam));
				break;

			case WM_LBUTTONUP:
				if (mouse_input)
					mouse_input->buttonUp(::GL::platform::Button::LEFT, LOWORD(lParam), HIWORD(lParam));
				break;

			case WM_MBUTTONUP:
				if (mouse_input)
					mouse_input->buttonUp(::GL::platform::Button::MIDDLE, LOWORD(lParam), HIWORD(lParam));
				break;

			case WM_RBUTTONUP:
				if (mouse_input)
					mouse_input->buttonUp(::GL::platform::Button::RIGHT, LOWORD(lParam), HIWORD(lParam));
				break;

			case WM_MOUSEMOVE:
				if (mouse_input)
					mouse_input->mouseMove(LOWORD(lParam), HIWORD(lParam));
				break;

			case WM_MOUSEWHEEL:
				if (mouse_input)
					mouse_input->mouseWheel(GET_WHEEL_DELTA_WPARAM(wParam));
				break;

			case WM_SYSCOMMAND:
				if (lParam == VK_RETURN)
					toggleFullscreen();
				else
					return DefWindowProcW(hwnd, msg, wParam, lParam);
				break;

			case WM_KEYDOWN:
				if (keyboard_input && (lParam & (1U << 30U)) == 0)
					keyboard_input->keyDown(keycode(wParam, lParam));
				break;

			case WM_KEYUP:
				if (keyboard_input)
					keyboard_input->keyUp(keycode(wParam, lParam));
				break;

			case WM_GETMINMAXINFO:
			{
				MINMAXINFO* info = reinterpret_cast<MINMAXINFO*>(lParam);
				info->ptMaxTrackSize.x = std::numeric_limits<LONG>::max();
				info->ptMaxTrackSize.y = std::numeric_limits<LONG>::max();
			}
				break;

			case WM_SIZE:
				if (display_handler)
					display_handler->resize(LOWORD(lParam), HIWORD(lParam));
				break;

			case WM_MOVE:
				if (display_handler)
					display_handler->move(static_cast<short>(LOWORD(lParam)), static_cast<short>(HIWORD(lParam)));
				break;

			case WM_ERASEBKGND:
				return 1;

			case WM_PAINT:
				ValidateRect(hWnd, 0);
				return 0;

			default:
				return DefWindowProcW(hWnd, msg, wParam, lParam);
			}

			return 0;
		}
	}
}
