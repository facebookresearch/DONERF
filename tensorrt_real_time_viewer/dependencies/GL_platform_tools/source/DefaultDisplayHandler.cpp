


#include <GL/platform/Application.h>
#include <GL/platform/DefaultDisplayHandler.h>


namespace GL
{
	namespace platform
	{
		void DefaultDisplayHandler::close()
		{
			GL::platform::quit();
		}

		void DefaultDisplayHandler::destroy()
		{
		}

		void DefaultDisplayHandler::move(int x, int y)
		{
		}

		void DefaultDisplayHandler::resize(int width, int height)
		{
		}
	}
}
