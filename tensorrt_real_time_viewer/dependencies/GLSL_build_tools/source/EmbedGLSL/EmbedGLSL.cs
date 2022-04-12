using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Diagnostics;
using Microsoft.Build.Framework;
using Microsoft.Build.Utilities;


namespace EmbedGLSL
{
	public class glsl2cpp : Task, ICancelableTask
	{
		[Required]
		public ITaskItem[] SourceFiles
		{
			get { return src_files; }
			set { src_files = value; }
		}

		[Output]
		public ITaskItem[] Outputs
		{
			get { return output_files; }
		}

		ITaskItem[] src_files;
		ITaskItem[] output_files;

		bool force_rebuild = false;
		public bool ForceRebuild
		{
			get { return force_rebuild; }
			set { force_rebuild = value; }
		}

		ITaskItem dependency_log;
		public ITaskItem DependencyLog
		{
			get { return dependency_log; }
			set { dependency_log = value; }
		}

		ITaskItem command_log;
		public ITaskItem CommandLog
		{
			get { return command_log; }
			set { command_log = value; }
		}

		int Compile(String cmdline, LogListener log_listener)
		{
			Log.LogMessageFromText("glsl2cpp " + cmdline, MessageImportance.Normal);
			Log.LogCommandLine(MessageImportance.Normal, "glsl2cpp " + cmdline);

			ProcessStartInfo sinfo = new ProcessStartInfo("glsl2cpp", cmdline);

			sinfo.UseShellExecute = false;
			sinfo.RedirectStandardError = true;
			sinfo.RedirectStandardOutput = true;

			Process process = Process.Start(sinfo);

			log_listener.Attach(process);

			process.WaitForExit();
			int exit_code = process.ExitCode;
			return exit_code;
		}

		volatile bool cancelled;

		public void Cancel()
		{
			cancelled = true;
		}

		public override bool Execute()
		{
			if (src_files == null || src_files.Length == 0)
				return true;

			bool success = true;

			List<ITaskItem> outputs = new List<ITaskItem>();

			String dependency_log_file = dependency_log.ItemSpec != null ? dependency_log.ItemSpec : null;
			String command_log_file = command_log.ItemSpec != null ? command_log.ItemSpec : null;

			Dependencies depends = new Dependencies(dependency_log_file, command_log_file);

			cancelled = false;

			for (int i = 0; i < src_files.Length && !cancelled; ++i)
			{
				ITaskItem item = src_files[i];
				String source_file = item.ItemSpec;

				String target_file = item.GetMetadata("TargetFile");

				if (String.IsNullOrWhiteSpace(target_file))
				{
					target_file = source_file + ".cpp";
					Log.LogWarning("no target file specified, defaulting to '" + target_file + '\'');
				}

				ITaskItem item_out = new TaskItem(target_file);
				item.CopyMetadataTo(item_out);

				CommandLine cmdline = new CommandLine(item, source_file, target_file);

				String target_title = source_file;

				if (depends.NeedsToBuild(force_rebuild, source_file, target_file, cmdline.CmdLineString))
				{
					Log.LogMessageFromText(source_file, MessageImportance.High);

					depends.ResetDependencies(source_file);

					LogListener log_listener = new LogListener(Log, cmdline, depends);

					if (Compile(cmdline.CmdLineString, log_listener) != 0)
					{
						success = false;
						continue;
					}
				}
				else
					Log.LogMessageFromText(target_title + " up to date", MessageImportance.Normal);

				outputs.Add(item_out);
				depends.SetCommandline(target_file, cmdline.CmdLineString);
			}

			output_files = outputs.ToArray();

			//depends.Write(dependency_log_file, command_log_file);

			return success;
		}
	}
}
