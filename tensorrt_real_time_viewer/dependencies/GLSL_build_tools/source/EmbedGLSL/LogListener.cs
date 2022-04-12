using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using Microsoft.Build.Framework;
using Microsoft.Build.Utilities;


namespace EmbedGLSL
{
	class LogListener
	{
		TaskLoggingHelper log;
		Dependencies depends;
		CommandLine cmdline;

		public LogListener(TaskLoggingHelper log, CommandLine cmdline, Dependencies dependencies)
		{
			this.log = log;
			this.depends = dependencies;
			this.cmdline = cmdline;
		}

		void StdOutCallback(object sender, DataReceivedEventArgs e)
		{
			if (e.Data != null)
			{
				if (String.Compare(e.Data, 0, "Opening file", 0, 12) == 0 ||
						String.Compare(e.Data, 0, "Current working dir", 0, 19) == 0)
					return;
				else if (String.Compare(e.Data, 0, "Resolved to", 0, 11) == 0)
				{
					String filename = e.Data.Substring(13, e.Data.Length - 14);
					depends.AddDependency(cmdline.SourceFile, filename);
				}
				else
					log.LogMessageFromText(e.Data, MessageImportance.High);
			}
		}

		void StdOutCallbackVi(object sender, DataReceivedEventArgs e)
		{
			if (e.Data != null)
			{
				if (String.Compare(e.Data, 0, "Resolved to", 0, 11) == 0)
				{
					String filename = e.Data.Substring(13, e.Data.Length - 14);
					depends.AddDependency(cmdline.SourceFile, filename);
				}

				log.LogMessageFromText(e.Data, MessageImportance.High);
			}
		}

		void StdErrCallback(object sender, DataReceivedEventArgs e)
		{
			if (e.Data != null)
				log.LogMessageFromText(e.Data, MessageImportance.High);
		}

		public void Attach(Process process)
		{
			if (cmdline.Vi)
				process.OutputDataReceived += StdOutCallbackVi;
			else
				process.OutputDataReceived += StdOutCallback;

			process.ErrorDataReceived += StdErrCallback;

			process.BeginOutputReadLine();
			process.BeginErrorReadLine();
		}
	}
}
