


#include "error.h"
#include "unicode.h"


namespace
{
	std::basic_string<WCHAR> utf8_to_utf16(const char* input, int input_length)
	{
		int output_length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, input, input_length, nullptr, 0);

		if (output_length <= 0)
			throw Win32::error(GetLastError());

		std::basic_string<WCHAR> output(output_length, WCHAR());

		if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, input, input_length, &output[0], output_length) <= 0)
			throw Win32::error(GetLastError());

		return output;
	}

	std::string utf_16_to_utf8(const WCHAR* input, int input_length)
	{
		int output_length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, input, input_length, nullptr, 0, nullptr, nullptr);

		if (output_length <= 0)
			throw Win32::error(GetLastError());

		std::string output(output_length, char());

		if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, input, input_length, &output[0], output_length, nullptr, nullptr) <= 0)
			throw Win32::error(GetLastError());

		return output;
	}
}

namespace Win32
{
	std::basic_string<WCHAR> widen(const char* string, size_t length)
	{
		return utf8_to_utf16(string, static_cast<int>(length));
	}
	
	std::basic_string<WCHAR> widen(const char* string)
	{
		auto str = utf8_to_utf16(string, -1);
		str.resize(str.length() - 1);
		return str;
	}
	
	std::basic_string<WCHAR> widen(const std::string& string)
	{
		return utf8_to_utf16(&string[0], static_cast<int>(string.length()));
	}
	
	
	std::string narrow(const WCHAR* string, size_t length)
	{
		return utf_16_to_utf8(string, static_cast<int>(length));
	}
	
	std::string narrow(const WCHAR* string)
	{
		auto str = utf_16_to_utf8(string, -1);
		str.resize(str.length() - 1);
		return str;
	}
	
	std::string narrow(const std::basic_string<WCHAR>& string)
	{
		return utf_16_to_utf8(&string[0], static_cast<int>(string.length()));
	}
}
