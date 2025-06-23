using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;

namespace ImageSegmentation.Converters
{
    internal class IntToBoolConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            int v = 0;
            int.TryParse(value.ToString(), out v);
            return v == 0 ? false : true;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            bool rev = System.Convert.ToBoolean(value);
            if (rev == true)
                return 1;
            else
                return 0;
        }
    }
}

