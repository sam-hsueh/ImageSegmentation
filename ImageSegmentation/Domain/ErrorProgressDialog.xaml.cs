using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace ImageSegmentation.Domain
{
    /// <summary>
    /// ErrorProgressDialog.xaml 的交互逻辑
    /// </summary>
    public partial class ErrorProgressDialog : UserControl
    {
        public ErrorProgressDialog(string message)
        {
            InitializeComponent();
            Message.Text = message + "......";
        }
        public string NewMessage
        {
            set
            {
                Message.Text = value + "......";
            }
        }
    }
}
