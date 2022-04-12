


#include <cstring>
#include <cctype>

#include <algorithm>
#include <iterator>

#include <string>
#include <vector>

#include <stdexcept>

#include <iostream>
#include <fstream>
#include <streambuf>


namespace
{
	const char* filename(const char* file_name)
	{
		const char* name = file_name + std::strlen(file_name);

		while (name > file_name && *(name - 1) != '/' && *(name - 1) != '\\')
			--name;

		return name;
	}

	const char* extension(char* file_name)
	{
		char* end = file_name + std::strlen(file_name);
		char* ext = end;

		while (ext != file_name && *--ext != '.')
			;

		if (ext != file_name)
		{
			std::transform(ext, end, ext, tolower);
			return ext;
		}

		return nullptr;
	}

	std::string directory(const char* file_name)
	{
		const char* name = file_name + std::strlen(file_name);

		while (name > file_name && *(name - 1) != '/' && *(name - 1) != '\\')
			--name;

		return std::string(file_name, name);
	}

	std::string findInclude(const char* include) { return ""; }

	std::istream& skipWhitespace(std::istream& file)
	{
		while (file && (file.peek() == ' ' || file.peek() == '\t'))
			file.get();
		return file;
	}

	std::istream& skipComment(std::istream& file)
	{
		if (file.peek() == '/')
			while (file && file.get() != '\n')
				;
		return file;
	}

	struct Options
	{
		const char* source;
		const char* target;
		std::vector<std::string> include_directories;
		std::vector<std::string> preprocessor_definitions;

		void addDefinition(const char* definition)
		{
			size_t length = std::strlen(definition);
			std::string d(definition, definition + length);
			std::replace_copy(definition, definition + length, &d[0], '=', ' ');
			preprocessor_definitions.push_back(std::move(d));
		}

		void addIncludeDirectory(const char* path)
		{
			include_directories.push_back(path);
			if (include_directories.back().back() != '/')
				include_directories.back().push_back('/');
		}
	};

	int next_source_string_number = 1;

	std::ostream& preprocessShaderSource(std::istream& in,
	                                     std::ostream& out,
	                                     const char* source,
	                                     const Options& options,
	                                     int depth = 0,
	                                     int source_string_number = 0);

	std::ostream& include(std::istream& in,
	                      std::ostream& out,
	                      const char* source,
	                      const Options& options,
	                      int depth,
	                      int source_string_number)
	{
		out << "\\n\"";
		preprocessShaderSource(
		    in, out, source, options, depth + 1, next_source_string_number++);
		out << "  \"";
		return out;
	}

	std::ostream& includeLocal(std::ostream& out,
	                           const char* include_string,
	                           const char* source_file,
	                           const Options& options,
	                           int depth,
	                           int source_string_number)
	{
		std::string file_name = directory(source_file) + include_string;

		std::ifstream in(file_name);

		if (in)
		{
			include(in,
			        out,
			        file_name.c_str(),
			        options,
			        depth + 1,
			        source_string_number);
			return out;
		}

		throw std::runtime_error(std::string("couldn't find include file '") +
		                         include_string + '\'');
	}

	std::ostream& includeGlobal(std::ostream& out,
	                            const char* include_string,
	                            const char* source_file,
	                            const Options& options,
	                            int depth,
	                            int source_string_number)
	{
		for (auto d = begin(options.include_directories);
		     d != end(options.include_directories);
		     ++d)
		{
			std::string file_name = *d + include_string;
			std::ifstream in(file_name);

			if (in)
			{
				include(in,
				        out,
				        file_name.c_str(),
				        options,
				        depth + 1,
				        source_string_number);
				return out;
			}
		}

		return includeLocal(out,
		                    include_string,
		                    source_file,
		                    options,
		                    depth,
		                    source_string_number);
	}

	std::ostream& preprocessingDirective(std::istream& in,
	                                     std::ostream& out,
	                                     const char* source_file,
	                                     const Options& options,
	                                     int depth,
	                                     int source_string_number)
	{
		std::string directive;
		in >> directive;
		if (directive == "include")
		{
			in >> skipWhitespace;
			char c = in.get();
			switch (c)
			{
				case '<':
				{
					std::string file;
					std::getline(in, file, '>');
					includeGlobal(out,
					              file.c_str(),
					              source_file,
					              options,
					              depth,
					              source_string_number);
					break;
				}

				case '"':
				{
					std::string file;
					std::getline(in, file, '"');
					includeLocal(out,
					             file.c_str(),
					             source_file,
					             options,
					             depth,
					             source_string_number);
					break;
				}

				default:
					throw std::runtime_error(
					    std::string("invalid include directive: ") + source_file);
			}
		}
		else if (directive == "version")
			throw std::runtime_error("multiple #version directives");
		else
			out << '#' << directive;

		return out;
	}

	std::ostream& preprocessShaderSource(std::istream& in,
	                                     std::ostream& out,
	                                     const char* source_file,
	                                     const Options& options,
	                                     int depth,
	                                     int source_string_number)
	{
		if (depth > 10)
			throw std::runtime_error("maximum include depth exceeded");

		out << "  \"#line 0 " << source_string_number << "\\n\"\n  \"";

		for (int c; (c = in.get()) != -1;)
		{
			switch (c)
			{
				case '/':
					if (in.peek() == '/')
						while (in && in.get() != '\n')
							;
					else if (in.peek() == '*')
					{
						in.get();
						while (in)
						{
							int c = in.get();
							if (c == '*' && in.peek() == '/')
							{
								in.get();
								break;
							}
						}
					}
					else
						out.put(c);
					break;

				case '#':
				{
					preprocessingDirective(
					    in, out, source_file, options, depth, source_string_number);
					break;
				}

				case '\n':
					out << "\\n\"\n  \"";
					break;

				case '"':
					out << "\\\"";
					break;

				case '\r':
					break;

				default:
					out.put(c);
					break;
			}
		}

		out << "\\n\"";

		return out;
	}

	std::string variableName(const char* file_name)
	{
		std::string name = filename(file_name);

		auto nend = name.rfind(".glsl");
		if (nend != std::string::npos)
			name = name.substr(0, nend);

		std::replace(begin(name), end(name), '.', '_');

		return name;
	}

	std::istream& skipToVersion(std::istream& in, std::ostream& out)
	{
		while (in && in.get() != '#');
		std::string directive;
		in >> directive;
		if (directive != "version")
			throw std::runtime_error("shader must start with #version directive");

		if (!in)
			throw std::runtime_error("no #version directive found");

		out << "  \"#version";
		while (in.peek() != '\n')
			out.put(in.get());
		out << "\\n\"\n";

		return in;
	}

	int run(const Options& options)
	{
		std::ifstream in(options.source);

		if (!in)
			throw std::runtime_error(std::string("couldn't open source file '") +
			                         options.source + '\'');

		std::ofstream out(options.target);

		if (!out)
			throw std::runtime_error(std::string("couldn't open target file '") +
			                         options.target + '\'');

		std::string var_name = variableName(options.source);

		out << "extern const char " << var_name << "[] = \n";
		for (auto d = begin(options.preprocessor_definitions);
		     d != end(options.preprocessor_definitions);
		     ++d)
			out << "  \"#define " << *d << "\\n\"\n";
		// out << "  \"layout(std140, row_major) uniform;\\n\"\n";
		skipToVersion(in, out);
		preprocessShaderSource(in, out, options.source, options);
		out << ";\n";

		return 0;
	}

	int version()
	{
		std::cout << "GLSL tool 0.0\n"
		             "by Michael Kenzel 2013" << std::endl;
		return 0;
	}

	int help()
	{
		std::cout << "usage: glsl {[-I <include directory>]|[-D<preprocessor "
		             "definition>]|[-o <target>]} <source>" << std::endl;
		return 0;
	}
}

int main(int argc, char* argv[])
{
	try
	{
		Options options;
		options.source = nullptr;
		options.target = nullptr;

		std::cout << "glsl ";
		for (int i = 1; i < argc; ++i)
			std::cout << "\"" << argv[i] << "\" ";
		std::cout << std::endl;

		size_t consumed = 0;
		for (int i = 1; i < argc; consumed = i++)
		{
			const char* a = argv[i];
			if (argv[i][0] == '-')
			{
				const char* option = &argv[i][1];

				if (option[0] == 'D')
				{
					if (option[1] == '\0')
					{
						if (argc < i + 2)
							throw std::runtime_error(
							    "expected preprocessor definition");
						options.addDefinition(argv[i + 1]);
					}
					else
						options.addDefinition(&option[1]);
				}
				else if (option[0] == 'I')
				{
					if (option[1] == '\0')
					{
						if (argc < i + 2)
							throw std::runtime_error("expected include directory");
						options.addIncludeDirectory(argv[i + 1]);
					}
					else
						options.addIncludeDirectory(&option[1]);
				}
				else if (strcmp(option, "o") == 0)
				{
					if (argc < i + 2)
						throw std::runtime_error("expected output file");
					options.target = argv[i + 1];
					++i;
				}
				else if ((strcmp(option, "-help") == 0) ||
				         (strcmp(option, "?") == 0))
				{
					help();
					return 0;
				}
				else if (strcmp(option, "-version") == 0)
				{
					version();
					return 0;
				}
				else
					throw std::runtime_error(std::string("unknown option '-") +
					                         option + '\'');
			}
		}

		if (consumed == argc - 1)
			options.source = argv[argc - 1];

		if (options.target == nullptr)
			throw std::runtime_error("no target given");
		if (options.source == nullptr)
			throw std::runtime_error("no source given");

		return run(options);
	}
	catch (std::exception& e)
	{
		std::cout << "error: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown error" << std::endl;
	}

	return -1;
}
