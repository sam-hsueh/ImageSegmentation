using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace Utilities
{

    public class RichTextBoxWriter : TextWriter
    {
        RichTextBox richTextBox;
        Window mainWindow;
        public RichTextBoxWriter(RichTextBox _richTextBox)
        {
            this.richTextBox = _richTextBox;
            mainWindow = Application.Current.MainWindow;
        }

        public override Encoding Encoding
        {
            get { return Encoding.UTF8; }
        }

        public override void Write(string value)
        {
            mainWindow.Dispatcher.Invoke(new Action(() =>
            {
                this.richTextBox.AppendText(value);
            }));
        }

        public override void WriteLine(string value)
        {
            this.richTextBox.AppendText(value + "\r");
        }
    }
}
