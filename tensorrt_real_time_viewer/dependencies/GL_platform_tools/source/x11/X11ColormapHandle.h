


#ifndef INCLUDED_PLATFORM_X11_COLORMAP_HANDLE
#define INCLUDED_PLATFORM_X11_COLORMAP_HANDLE

#pragma once

#include <utility>

#include "platform.h"


namespace X11
{
	class ColormapHandle
	{
	private:
		ColormapHandle(const ColormapHandle&);
		ColormapHandle& operator =(const ColormapHandle&);

		::Display* display;
		::Colormap colormap;

	public:
		ColormapHandle()
			: display(nullptr),
			  colormap(0)
		{
		}

		ColormapHandle(::Display* display, ::Colormap colormap)
			: display(display),
			  colormap(colormap)
		{
		}

		ColormapHandle(ColormapHandle&& cm)
			: display(cm.display),
			  colormap(cm.colormap)
		{
			cm.colormap = 0;
		}

		~ColormapHandle()
		{
			if (colormap)
				XFreeColormap(display, colormap);
		}

		ColormapHandle& operator =(ColormapHandle&& cm)
		{
			using std::swap;
			display = cm.display;
			swap(colormap, cm.colormap);
			return *this;
		}

		operator Colormap() const { return colormap; }
	};

	inline ColormapHandle createColorMap(::Display* display, ::Window window, ::Visual* visual, int alloc)
	{
		return ColormapHandle(display, XCreateColormap(display, window, visual, alloc));
	}
}

#endif  // INCLUDED_PLATFORM_X11_COLORMAP_HANDLE
