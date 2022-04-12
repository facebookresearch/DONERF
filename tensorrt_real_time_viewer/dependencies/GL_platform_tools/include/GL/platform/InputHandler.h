


#ifndef INCLUDED_GL_PLATFORM_INPUT_HANDLER
#define INCLUDED_GL_PLATFORM_INPUT_HANDLER

#pragma once

#include "interface.h"

#if defined(_WIN32)
#include "../../../source/win32/Win32Input.h"
#elif defined(__gnu_linux__)
#include "../../../source/x11/X11Input.h"
#else
#error "platform not supported."
#endif


namespace GL
{
	namespace platform
	{
		class INTERFACE MouseInputHandler
		{
		protected:
			MouseInputHandler() {}
			MouseInputHandler(const MouseInputHandler&) {}
			MouseInputHandler& operator =(const MouseInputHandler&) { return *this; }
			~MouseInputHandler() {}
		public:
			virtual void buttonDown(Button button, int x, int y) = 0;
			virtual void buttonUp(Button button, int x, int y) = 0;
			virtual void mouseMove(int x, int y) = 0;
			virtual void mouseWheel(int delta) = 0;
		};

		class INTERFACE KeyboardInputHandler
		{
		protected:
			KeyboardInputHandler() {}
			KeyboardInputHandler(const KeyboardInputHandler&) {}
			KeyboardInputHandler& operator =(const KeyboardInputHandler&) { return *this; }
			~KeyboardInputHandler() {}
		public:
			virtual void keyDown(Key key) = 0;
			virtual void keyUp(Key key) = 0;
		};

		class INTERFACE ConsoleHandler
		{
		protected:
			ConsoleHandler() {}
			ConsoleHandler(const ConsoleHandler&) {}
			ConsoleHandler& operator =(const ConsoleHandler&) { return *this; }
			~ConsoleHandler() {}
		public:
			virtual void command(const char* command, size_t length) = 0;
		};
	}
}

#endif  // INCLUDED_GL_PLATFORM_INPUT_HANDLER
