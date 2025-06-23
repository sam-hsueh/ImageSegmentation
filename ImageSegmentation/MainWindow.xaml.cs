using MaterialDesignColors;
using ImageSegmentation.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Controls.Primitives;

namespace ImageSegmentation
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        ThemeSettingsViewModel? tsv;
        MainWindowViewModel? mvm;
        int themeId = 0;
        bool isDarkTheme = false;
        List<Swatch> Swatches;
        Home home;

        public MainWindow()
        {
            InitializeComponent();
            tsv = new ThemeSettingsViewModel();
            Swatches = tsv.Swatches.ToList();
            Closing += MainWindow_Closing;
            //Loaded += MainWindow_Loaded;
            themeId = Properties.Settings.Default.ThemeId;
            isDarkTheme = Properties.Settings.Default.DarkTheme;
            Swatches = tsv.Swatches.ToList();
            tsv.ApplyPrimaryCommand.Execute(Swatches[themeId]);
            tsv.IsDarkTheme = isDarkTheme;
            DataContext = new MainWindowViewModel();
            mvm = (MainWindowViewModel)DataContext;
            Home? c = mvm?.MenuItems[0].Content as Home;
            home = c;
            //RoboConn.Background = System.Windows.Media.Brushes.Transparent;
            //LaserConn.Background = System.Windows.Media.Brushes.Transparent;
            AddHotKeys();
        }
        private void MainWindow_Closing(object? sender, System.ComponentModel.CancelEventArgs e)
        {
            Properties.Settings.Default.ThemeId = themeId;
            Properties.Settings.Default.InitDir = mvm!.OriginalImageDir;
            Properties.Settings.Default.Save();
            GC.Collect();
            Application.Current.Shutdown();
            Environment.Exit(0);
        }

        private void UIElement_OnPreviewMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            var dependencyObject = Mouse.Captured as DependencyObject;

            while (dependencyObject != null)
            {
                if (dependencyObject is ScrollBar) return;
                dependencyObject = VisualTreeHelper.GetParent(dependencyObject);
            }
        }
        private void AddHotKeys()
        {
            try
            {
                RoutedCommand Settings1 = new RoutedCommand();
                Settings1.InputGestures.Add(new KeyGesture(Key.Q, ModifierKeys.Control));
                CommandBindings.Add(new CommandBinding(Settings1, My_first_event_handler));

                RoutedCommand Settings2 = new RoutedCommand();
                Settings2.InputGestures.Add(new KeyGesture(Key.A, ModifierKeys.Control));
                CommandBindings.Add(new CommandBinding(Settings2, My_second_event_handler));
                RoutedCommand Settings3 = new RoutedCommand();
                Settings3.InputGestures.Add(new KeyGesture(Key.D, ModifierKeys.Control));
                CommandBindings.Add(new CommandBinding(Settings3, My_third_event_handler));
                RoutedCommand Settings6 = new RoutedCommand();
                //Settings6.InputGestures.Add(new KeyGesture(Key.M, ModifierKeys.Control));
                //CommandBindings.Add(new CommandBinding(Settings6, My_event_handler6));

            }
            catch (Exception err)
            {
                throw new Exception(err.ToString());
            }
        }
        private void My_first_event_handler(object sender, ExecutedRoutedEventArgs e)
        {
            themeId++;
            themeId = themeId >= Swatches.Count ? 0 : themeId;
            tsv?.ApplyPrimaryCommand.Execute(Swatches[themeId]);
            Properties.Settings.Default.ThemeId = themeId;
            Properties.Settings.Default.Save();
            //var mvm = DataContext as MainWindowViewModel;
            //Home? c = mvm?.MenuItems[0].Content as Home;
            //home = c;
            ////home?.Render();
        }

        private void My_second_event_handler(object sender, ExecutedRoutedEventArgs e)
        {
            themeId--;
            themeId = themeId < 0 ? Swatches.Count - 1 : themeId;
            tsv?.ApplyPrimaryCommand.Execute(Swatches[themeId]);
            Properties.Settings.Default.ThemeId = themeId;
            Properties.Settings.Default.Save();
            //var mvm = DataContext as MainWindowViewModel;
            //Home? c = mvm?.MenuItems[0].Content as Home;
            //home = c;
            //home?.Render();
        }
        //private void My_event_handler6(object sender, ExecutedRoutedEventArgs e)
        //{
        //    if (PCIDataViewModel.Simu)
        //        PCIDataViewModel.Simu = false;
        //    else
        //        PCIDataViewModel.Simu = true;
        //}
        private void My_third_event_handler(object sender, ExecutedRoutedEventArgs e)
        {
            if (isDarkTheme == true)
                isDarkTheme = false;
            else
                isDarkTheme = true;
            tsv!.IsDarkTheme = isDarkTheme;
            Properties.Settings.Default.DarkTheme = tsv!.IsDarkTheme;
            Properties.Settings.Default.Save();
            //if (tsv?.IsDarkTheme ?? false)
            //    tsv.IsDarkTheme = false;
            //else
            //    tsv!.IsDarkTheme = true;
            //var mvm = DataContext as MainWindowViewModel;
            //Home? c = mvm?.MenuItems[0].Content as Home;
            //home = c;
            ////home?.Render();
            //var paletteHelper = new PaletteHelper();
            //var theme = paletteHelper.GetTheme();

            //theme.SetBaseTheme(theme.GetBaseTheme() == BaseTheme.Light ? Theme.Dark : Theme.Light);
            //paletteHelper.SetTheme(theme);
        }

        private void MenuItemsListBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if(mvm!.SelectedIndex==0)
            {
                home.ctime = 5;
            }
        }
        //private void OnSelectedItemChanged(object sender, DependencyPropertyChangedEventArgs e) => MainScrollViewer.ScrollToHome();
    }
}
