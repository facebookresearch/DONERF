using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Build.Framework;


namespace EmbedGLSL
{
	internal class CommandLineBuilder
	{
		StringBuilder cmd_line = new StringBuilder();

		public override string ToString()
		{
			return cmd_line.ToString();
		}

		public CommandLineBuilder()
		{
		}

		public bool Append(String s)
		{
			if (!String.IsNullOrWhiteSpace(s))
			{
				cmd_line.Append(s).Append(' ');
				return true;
			}
			return false;
		}

		public bool Append(String flag, String value)
		{
			if (!String.IsNullOrWhiteSpace(value))
			{
				cmd_line.Append(flag).Append(value).Append(' ');
				return true;
			}
			return false;
		}

		public bool AppendBoolFlag(String flag, String value)
		{
			if (!String.IsNullOrWhiteSpace(value) && value == "true")
			{
				cmd_line.Append(flag).Append(' ');
				return true;
			}
			return false;
		}

		public bool AppendStringList(String list)
		{
			if (!String.IsNullOrWhiteSpace(list))
			{
				String[] items = list.Split(';');
				foreach (String s in items)
					if (s.Length > 0)
						cmd_line.Append(' ').Append(s).Append(' ');
				return true;
			}
			return false;
		}

		public bool AppendStringList(String flag, String list)
		{
			if (!String.IsNullOrWhiteSpace(list))
			{
				String[] items = list.Split(';');
				foreach (String s in items)
					if (s.Length > 0)
						cmd_line.Append(flag).Append(s).Append(' ');
				return true;
			}
			return false;
		}

		private static String fixPath(String path)
		{
			path = path.Trim();
			if (path[0] != '"')
			{
				for (int i = 0; i < path.Length; ++i)
				{
					if (Char.IsWhiteSpace(path[i]))
					{
						return '"' + path + '"';
					}
				}
			}
			return path;
		}

		public bool AppendFilename(String name)
		{
			if (!String.IsNullOrWhiteSpace(name))
			{
				cmd_line.Append(fixPath(name)).Append(' ');
				return true;
			}
			return false;
		}

		public bool AppendFilename(String flag, String name)
		{
			if (!String.IsNullOrWhiteSpace(name))
			{
				cmd_line.Append(flag).Append(fixPath(name)).Append(' ');
				return true;
			}
			return false;
		}

		public bool AppendFileList(String list)
		{
			return AppendDirectoryList(list);
		}

		public bool AppendFileList(String flag, String list)
		{
			return AppendDirectoryList(flag, list);
		}

		public bool AppendDirectoryList(String list)
		{
			if (!String.IsNullOrWhiteSpace(list))
			{
				String[] items = list.Split(';');
				foreach (String s in items)
					if (s.Length > 0)
						cmd_line.Append(fixPath(s)).Append(' ');
				return true;
			}
			return false;
		}

		public bool AppendDirectoryList(String flag, String list)
		{
			if (!String.IsNullOrWhiteSpace(list))
			{
				String[] items = list.Split(';');
				foreach (String s in items)
					if (s.Length > 0)
						cmd_line.Append(flag).Append(fixPath(s)).Append(' ');
				return true;
			}
			return false;
		}
	}

	class CommandLine
	{
		ITaskItem item;
		String source_file;
		String target_file;
		String preprocess_cmdline;
		String cmd_line;
		bool vi;

		public ITaskItem Item
		{
			get { return item; }
		}

		public String SourceFile
		{
			get { return source_file; }
		}

		public String TargetFile
		{
			get { return target_file; }
		}

		public String CmdLineString
		{
			get { return cmd_line; }
		}

		public String PreprocessCmdLineString
		{
			get { return preprocess_cmdline; }
		}

		public bool Vi
		{
			get { return vi; }
		}

		public CommandLine(ITaskItem item, String source_file, String target_file)
		{
			this.item = item;
			this.source_file = source_file;
			this.target_file = target_file;
			this.preprocess_cmdline = null;
			this.vi = false;

			CommandLineBuilder cmd_line = new CommandLineBuilder();

			cmd_line.AppendDirectoryList("-I ", item.GetMetadata("AdditionalIncludeDirectories"));
			cmd_line.AppendStringList("-D", item.GetMetadata("PreprocessorDefinitions"));

			cmd_line.AppendFilename("-o ", target_file);
			cmd_line.AppendFilename(source_file);

			this.cmd_line = cmd_line.ToString();
		}
	}
}
