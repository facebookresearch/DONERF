


#include <atomic>
#include <unordered_map>
#include <iostream>

#include "X11Display.h"
#include "X11GLWindow.h"
#include "X11GLApplication.h"


namespace
{
	std::atomic<bool> run_mainloop;
}

namespace X11
{
	namespace GL
	{
		X11::Display display = X11::openDisplay();
		extern std::unordered_map< ::Window, X11::GL::Window*> window_map;

		void run(::GL::platform::Renderer& renderer)
		{
			run(renderer, nullptr);
		}

		void run(::GL::platform::Renderer& renderer, ::GL::platform::ConsoleHandler* console_handler)
		{
			XEvent event;
			run_mainloop = true;

			while (run_mainloop)
			{
				while (XPending(display) > 0)
				{
					XNextEvent(display, &event);

					if (X11::GL::Window* window = window_map[event.xany.window])
						window->handleEvent(event);
				}

				renderer.render();
			}
		}

		void quit()
		{
			run_mainloop = false;
		}
	}
}
