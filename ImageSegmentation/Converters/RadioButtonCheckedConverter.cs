using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;
using ImageSegmentation.Domain;

namespace ImageSegmentation.Converters
{
    public class RadioButtonCheckedConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter,System.Globalization.CultureInfo culture)
        {
            int v = -1;
            var t = value.GetType();
            if (t == typeof(YoloType))
            {
                v = (int)value;
            }
            else if (t == typeof(DeviceType)) 
            {
                v = (int)value;
            }
            else if (t == typeof(ScalarType))
            {
                v = (int)value;
            }
            else if (t == typeof(ModelType))
            {
                v = (int)value;
            }
            return v.Equals(System.Convert.ToInt32(parameter)); ;
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            return value.Equals(true) ? parameter : Binding.DoNothing;
        }
    }
}

