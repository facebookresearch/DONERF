using System;
using System.Collections.Generic;
using System.Text;
using System.IO;


namespace EmbedGLSL
{
	class Dependencies
	{
		Dictionary<String, HashSet<String>> dependencies = new Dictionary<String, HashSet<String>>();
		Dictionary<String, String> commandlines = new Dictionary<String, String>();

		static void ReadLog(String filename, Dictionary<String, HashSet<String>> dest)
		{
			using (FileStream file = new FileStream(filename, FileMode.Open, FileAccess.Read))
			{
				using (StreamReader log = new StreamReader(file))
				{
					String line;
					HashSet<String> current = null;

					while ((line = log.ReadLine()) != null)
					{
						if (line.Length > 0)
						{
							if (line[0] == '^')
							{
								current = new HashSet<String>();
								dest.Add(line.Substring(1), current);
							}
							else if (current != null)
								current.Add(line);
						}
					}
				}
			}
		}

		static void ReadLog(String filename, Dictionary<String, String> dest)
		{
			using (FileStream file = new FileStream(filename, FileMode.Open, FileAccess.Read))
			{
				using (StreamReader log = new StreamReader(file))
				{
					String line;
					while ((line = log.ReadLine()) != null)
					{
						if (line.Length > 0)
						{
							if (line[0] == '^')
							{
								String l = log.ReadLine();
								if (l != null)
									dest.Add(line.Substring(1), l);
							}
						}
					}
				}
			}
		}

		static void WriteLog(String filename, Dictionary<String, HashSet<String>> src)
		{
			using (FileStream file = new FileStream(filename, FileMode.Create, FileAccess.Write))
			{
				using (StreamWriter log = new StreamWriter(file))
				{
					foreach (KeyValuePair<String, HashSet<String>> d in src)
					{
						log.WriteLine('^' + d.Key);
						foreach (String dep in d.Value)
							log.WriteLine(dep);
					}
				}
			}
		}

		static void WriteLog(String filename, Dictionary<String, String> src)
		{
			using (FileStream file = new FileStream(filename, FileMode.Create, FileAccess.Write))
			{
				using (StreamWriter log = new StreamWriter(file))
				{
					foreach (KeyValuePair<String, String> d in src)
					{
						log.WriteLine('^' + d.Key);
						log.WriteLine(d.Value);
					}
				}
			}
		}


		public Dependencies(String dependencylog, String commandlog)
		{
			try
			{
				if (dependencylog != null)
					ReadLog(dependencylog, dependencies);
				if (commandlog != null)
					ReadLog(commandlog, commandlines);
			}
			catch (IOException)
			{
			}
		}

		public bool NeedsToBuild(bool force_rebuild, String source, String target, String cmdline)
		{
			try
			{
				if (force_rebuild)
					return true;

				source = Path.GetFullPath(source);
				target = Path.GetFullPath(target);

				DateTime target_stamp = File.GetLastWriteTime(target);

				if (target_stamp < File.GetLastWriteTime(source))
					return true;

				if (dependencies == null)
					return false;

				HashSet<String> deps;
				if (dependencies.TryGetValue(source, out deps))
				{
					foreach (String dep in deps)
						if (target_stamp < File.GetLastWriteTime(dep))
							return true;
				}

				String cmd_line;
				if (commandlines.TryGetValue(target, out cmd_line) && cmd_line == cmdline)
					return false;

				return true;
			}
			catch (Exception)
			{
				return true;
			}
		}

		public void AddDependency(String source, String dependency)
		{
			source = Path.GetFullPath(source);

			HashSet<String> deps;
			if (dependencies.TryGetValue(source, out deps))
			{
				deps.Add(dependency);
			}
			else
			{
				deps = new HashSet<String>();
				deps.Add(dependency);
				dependencies.Add(source, deps);
			}
		}

		public void ResetDependencies(String source)
		{
			source = Path.GetFullPath(source);

			HashSet<String> deps;
			if (dependencies.TryGetValue(source, out deps))
				deps.Clear();
		}

		public void SetCommandline(String target, String cmdline)
		{
			target = Path.GetFullPath(target);

			commandlines[target] = cmdline;
		}

		public void Write(String dependencylog, String commandlog)
		{
			if (dependencylog != null)
				WriteLog(dependencylog, dependencies);
			if (commandlog != null)
				WriteLog(commandlog, commandlines);
		}
	}
}
