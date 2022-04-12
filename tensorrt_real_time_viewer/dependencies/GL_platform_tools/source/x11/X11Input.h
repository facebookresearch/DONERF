


#ifndef INCLUDED_X11_GL_INPUT
#define INCLUDED_X11_GL_INPUT

#pragma once

#include "platform.h"


namespace GL
{
	namespace platform
	{
		enum class Button
		{
			LEFT = 1U,
			RIGHT = 4U,
			MIDDLE = 2U
		};

		enum class Key
		{
			SHIFT_L = XK_Shift_L,
			CTRL_L = XK_Control_L,
			ALT_L = XK_Alt_L,
			SHIFT_R = XK_Shift_R,
			CTRL_R = XK_Control_R,
			ALT_R = XK_Alt_R,
			SPACE = XK_space,
			ENTER = XK_Return,
			BACKSPACE = XK_BackSpace,
			TAB = XK_Tab,
			INS = XK_Insert,
			DEL = XK_Delete,
			HOME = XK_Home,
			LEFT = XK_Left,
			UP = XK_Up,
			RIGHT = XK_Right,
			DOWN = XK_Down,
			PAGE_UP = XK_Page_Up,
			PAGE_DOWN = XK_Page_Down,
			C_0 = XK_0,
			C_1,
			C_2,
			C_3,
			C_4,
			C_5,
			C_6,
			C_7,
			C_8,
			C_9,
			PLUS = XK_plus,
			MINUS = XK_minus,
			NUM_0 = XK_0,
			NUM_1,
			NUM_2,
			NUM_3,
			NUM_4,
			NUM_5,
			NUM_6,
			NUM_7,
			NUM_8,
			NUM_9,
			C_A = XK_a,
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
			F1 = XK_F1,
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
			ESCAPE = XK_Escape
		};
	}
}

#endif  // INCLUDED_X11_GL_INPUT
