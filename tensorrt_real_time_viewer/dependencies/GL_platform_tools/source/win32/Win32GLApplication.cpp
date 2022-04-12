#include <string>
#include <thread>
#include <atomic>

#include <win32/event.h>

#include "Win32GLApplication.h"


namespace
{
	static const UINT MSG_CONSOLE = WM_APP + 1;

	std::atomic<bool> run_console;
	Win32::unique_handle<HANDLE, 0, Win32::CloseHandleDeleter> command_processed_event;

	void console(DWORD main_thread)
	{
		HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
		HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);

		while (run_console)
		{
			DWORD bytes_written;
			WriteFile(out, "> ", 2, &bytes_written, nullptr);

			std::string line;

			do
			{
				static const size_t buffer_size = 1024;
				char buffer[buffer_size];

				DWORD bytes_read;
				ReadFile(in, buffer, buffer_size, &bytes_read, nullptr);

				for (size_t i = 0; i < bytes_read; ++i)
				{
					if (buffer[i] == '\n')
					{
						if (!line.empty() && line.back() == '\r')
							PostThreadMessageW(main_thread, MSG_CONSOLE, line.length() - 1, reinterpret_cast<LPARAM>(&line[0]));
						else
							PostThreadMessageW(main_thread, MSG_CONSOLE, line.length(), reinterpret_cast<LPARAM>(&line[0]));
						WaitForSingleObject(command_processed_event, INFINITE);
						line.clear();
					}
					else
						line += buffer[i];
				}
			} while (!line.empty());
		}
	}

	bool processMessages()
	{
		MSG msg;

		while (PeekMessageW(&msg, 0, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
				return false;

			TranslateMessage(&msg);
			DispatchMessageW(&msg);
		}

		return true;
	}

	bool processMessages(GL::platform::ConsoleHandler* console_handler)
	{
		MSG msg;

		while (PeekMessageW(&msg, 0, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
				return false;
			else if (msg.message == MSG_CONSOLE)
			{
				console_handler->command(reinterpret_cast<const char*>(msg.lParam), msg.wParam);
				SetEvent(command_processed_event);
			}
			else
			{
				TranslateMessage(&msg);
				DispatchMessageW(&msg);
			}
		}

		return true;
	}
}

namespace Win32
{
	namespace GL
	{
		void run(::GL::platform::Renderer& renderer)
		{
			while (processMessages())
				renderer.render();
		}

		void run(::GL::platform::Renderer& renderer, ::GL::platform::ConsoleHandler* console_handler)
		{
			SetConsoleCP(CP_UTF8);
			SetConsoleOutputCP(CP_UTF8);

			command_processed_event = Win32::createEvent();

			run_console.store(true);

			std::thread console_thread(console, GetCurrentThreadId());

			while (processMessages(console_handler))
			{
				renderer.render();
			}

			run_console.store(false);
			CancelSynchronousIo(console_thread.native_handle());
			console_thread.join();
		}

		void quit()
		{
			PostQuitMessage(0);
		}
	}
}
