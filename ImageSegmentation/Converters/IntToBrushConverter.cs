using ImageSegmentation.Domain;
using System;
using System.Globalization;
using System.Resources;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;

namespace ImageSegmentation.Converters
{
    [ValueConversion(typeof(int), typeof(Brush))]
    public sealed class IntToBrushConverter : IValueConverter
    {
        //Color[] a = new Color[] { (Color)ColorConverter.ConvertFromString("#FF3F51B5"), (Color)ColorConverter.ConvertFromString("#FF3A7E00"), (Color)ColorConverter.ConvertFromString("#FFB00020")};
        public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            // var dynamicResource = new DynamicResourceExtension("MaterialDesign.Brush.Primary");
            // Brush[] a =  {(System.Windows.Media.Brush)App.Current.FindResource("MaterialDesign.Brush.Primary"),
            //(System.Windows.Media.Brush)App.Current.FindResource("MaterialDesign.Brush.Secondary"),
            //(System.Windows.Media.Brush)App.Current.FindResource("MaterialDesign.Brush.ValidationError") };
            Color[] a = Home.colors!;
            if (a == null)
                return null;
            if (value is int color)
            {
//                SolidColorBrush rv = (SolidColorBrush)(a![color]);
                SolidColorBrush rv = new SolidColorBrush(a![color]);
                rv.Freeze();
                return rv;
            }
            return Binding.DoNothing;
        }

        public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            if (value is SolidColorBrush brush)
            {
                return 0;
            }
            return 0;
        }
    }
}
