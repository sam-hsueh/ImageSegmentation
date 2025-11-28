using System;
using System.Collections.Generic;
using System.Drawing;
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
using ImageSegmentation.Domain;

namespace ImageSegmentation
{
    /// <summary>
    /// Setting.xaml 的交互逻辑
    /// </summary>
    public partial class Settings : UserControl
    {
        MainWindowViewModel? mwv;
        public Settings()
        {
            InitializeComponent();
            Loaded += Settings_Loaded;
        }
        private void Settings_Loaded(object sender, RoutedEventArgs e)
        {
            Window? window = Window.GetWindow(this);
            mwv = this.DataContext as MainWindowViewModel;
            mwv!.setW = this;
        }        
    }
}
