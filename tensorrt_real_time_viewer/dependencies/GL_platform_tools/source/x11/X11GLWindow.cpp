


#include <cassert>
#include <atomic>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <iostream>

#include "X11GLApplication.h"
#include "X11GLWindow.h"


namespace
{
	GLXFBConfig findFBConfig(::Display* display, int depth_buffer_bits, int stencil_buffer_bits, bool stereo)
	{
		static int attribs[] = {
			GLX_X_RENDERABLE    , True,
			GLX_DRAWABLE_TYPE   , GLX_WINDOW_BIT,
			GLX_RENDER_TYPE     , GLX_RGBA_BIT,
			GLX_X_VISUAL_TYPE   , GLX_TRUE_COLOR,
			GLX_RED_SIZE        , 8,
			GLX_GREEN_SIZE      , 8,
			GLX_BLUE_SIZE       , 8,
			GLX_ALPHA_SIZE      , 8,
			GLX_DEPTH_SIZE      , depth_buffer_bits,
			GLX_STENCIL_SIZE    , stencil_buffer_bits,
			GLX_DOUBLEBUFFER    , True,
			GLX_STEREO          , stereo ? True : False,
			None
		};

		int num_configs;
		std::unique_ptr<GLXFBConfig[], X11::deleter> configs(glXChooseFBConfig(display, DefaultScreen(display), attribs, &num_configs));

		if (configs == nullptr)
			throw std::runtime_error("no matching GLXFBConfig.");

		return configs[0];
	}

	X11::WindowHandle createWindow(Display* display, int width, int height, XVisualInfo* vi, Colormap colormap)
	{
		XSetWindowAttributes swa;
		swa.colormap = colormap;
		swa.background_pixmap = None;
		swa.border_pixel = 0;
		swa.event_mask = StructureNotifyMask | KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask;

		return X11::createWindow(display, RootWindow(display, vi->screen), 0, 0, width, height, 0, vi->depth, InputOutput, vi->visual, CWBorderPixel | CWColormap | CWEventMask, &swa);
	}
}

namespace X11
{
	namespace GL
	{
		extern X11::Display display;
		std::unordered_map< ::Window, X11::GL::Window*> window_map;

		Window::Window(const char* title, int width, int height, int depth_buffer_bits, int stencil_buffer_bits, bool stereo)
			: fb_config(findFBConfig(display, depth_buffer_bits, stencil_buffer_bits, stereo)),
			  vi(glXGetVisualFromFBConfig(display, fb_config)),
			  colormap(createColorMap(display, DefaultRootWindow(static_cast< ::Display*>(display)), vi->visual, AllocNone)),
			  window(::createWindow(display, width, height, vi.get(), colormap)),
			  display_handler(nullptr),
			  keyboard_handler(nullptr),
			  mouse_handler(nullptr)
		{
			window_map[window] = this;

			Atom wmDelete = XInternAtom(display, "WM_DELETE_WINDOW", False);
			XSetWMProtocols(display, window, &wmDelete, 1);

			XMapWindow(display, window);

			Window::title(title);

			XSync(display, False);
		}

		Window::~Window()
		{
			window_map.erase(window);
		}

		void Window::title(const char* title)
		{
			XStoreName(display, window, title);
			Atom format_utf8 = XInternAtom(display, "UTF8_STRING", False);
			Atom net_wm_name = XInternAtom(display, "_NET_WM_NAME", False);
			XChangeProperty(display, window, net_wm_name, format_utf8, 8, PropModeReplace, reinterpret_cast<const unsigned char*>(title), std::strlen(title));
		}

		Context Window::createContext(int version_major, int version_minor, bool debug)
		{
			return X11::GL::createContext(display, fb_config, version_major, version_minor, debug);
		}

		void Window::attach(::GL::platform::DisplayHandler* handler)
		{
			display_handler = handler;
		}

		void Window::attach(::GL::platform::MouseInputHandler* handler)
		{
			mouse_handler = handler;
		}

		void Window::attach(::GL::platform::KeyboardInputHandler* handler)
		{
			keyboard_handler = handler;
		}

		void Window::handleEvent(const XEvent& event)
		{
			switch (event.type)
			{
				case DestroyNotify:
					if (display_handler)
						display_handler->destroy();
					break;

				case ButtonPress:
					if (mouse_handler)
						mouse_handler->buttonDown(static_cast< ::GL::platform::Button>(1 << (event.xbutton.button - 1)), event.xbutton.x, event.xbutton.y);
					break;

				case ButtonRelease:
					if (mouse_handler)
					{
						if (event.xbutton.button == 5)
						{
							mouse_handler->mouseWheel(3);
						}
						else if (event.xbutton.button == 4)
						{
							mouse_handler->mouseWheel(-3);
						}
						else
						{
						mouse_handler->buttonUp(static_cast< ::GL::platform::Button>(1 << (event.xbutton.button - 1)), event.xbutton.x, event.xbutton.y);
						}
          }
					break;

				case MotionNotify:
					if (mouse_handler)
						mouse_handler->mouseMove(event.xmotion.x, event.xmotion.y);
					break;

				case KeyPress:
					if (keyboard_handler)
						keyboard_handler->keyDown(static_cast< ::GL::platform::Key>(XkbKeycodeToKeysym(display, event.xkey.keycode, 0, 0)));
					break;

				case KeyRelease:
					if (keyboard_handler)
						keyboard_handler->keyUp(static_cast< ::GL::platform::Key>(XkbKeycodeToKeysym(display, event.xkey.keycode, 0, 0)));
					break;

				case ConfigureNotify:
					if (display_handler)
						display_handler->resize(event.xconfigure.width, event.xconfigure.height);
					break;

				case ClientMessage:
				{
					Atom wmDeleteMessage = XInternAtom(display, "WM_DELETE_WINDOW", false);
					if (event.xclient.data.l[0] == wmDeleteMessage)
					{
						if (display_handler)
							display_handler->close();
						else
							quit();
					}
					break;
				}
			}
		}
	}
}
