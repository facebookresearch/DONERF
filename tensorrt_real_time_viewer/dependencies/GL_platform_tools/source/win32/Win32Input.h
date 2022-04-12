


#ifndef INCLUDED_WIN32_GL_INPUT
#define INCLUDED_WIN32_GL_INPUT

#pragma once

#include <win32/platform.h>


namespace GL
{
	namespace platform
	{
		enum class Button
		{
			LEFT = VK_LBUTTON,
			RIGHT = VK_RBUTTON,
			MIDDLE = VK_MBUTTON
		};

		enum class Key : unsigned int
		{
			SHIFT = 0x10,
			CTRL = 0x11,
			ALT = 0x12,
			SPACE = 0x20,
			ENTER = 0x0D,
			BACKSPACE = 0x08,
			TAB = 0x09,
			INS = 0x2D,
			DEL = 0x2E,
			HOME = 0x24,
			LEFT = 0x25,
			UP = 0x26,
			RIGHT = 0x27,
			DOWN = 0x28,
			PAGE_UP = 0x21,
			PAGE_DOWN = 0x22,
			C_0 = 0x30,
			C_1,
			C_2,
			C_3,
			C_4,
			C_5,
			C_6,
			C_7,
			C_8,
			C_9,
			PLUS = 0x6B,
			MINUS = 0x6D,
			NUM_0 = 0x60,
			NUM_1,
			NUM_2,
			NUM_3,
			NUM_4,
			NUM_5,
			NUM_6,
			NUM_7,
			NUM_8,
			NUM_9,
			C_A = 0x41,
			C_B,
			C_C,
			C_D,
			C_E,
			C_F,
			C_G,
			C_H,
			C_I,
			C_J,
			C_K,
			C_L,
			C_M,
			C_N,
			C_O,
			C_P,
			C_Q,
			C_R,
			C_S,
			C_T,
			C_U,
			C_V,
			C_W,
			C_X,
			C_Y,
			C_Z,
			F1 = 0x70,
			F2,
			F3,
			F4,
			F5,
			F6,
			F7,
			F8,
			F9,
			F10,
			F11,
			F12,
			ESCAPE = 0x1B
		};
	}
}

#endif  // INCLUDED_WIN32_GL_INPUT
