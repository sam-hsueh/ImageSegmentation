using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Data;

namespace ImageSegmentation.Converters
{
    internal class Array2DValueConverter : IMultiValueConverter
    {
        #region interface implementations

        int index1 = -1, index2 = -1;
        ObservableCollection<double[]>? a;
        public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
        {
            a = values[1] as ObservableCollection<double[]>;
            index1 = System.Convert.ToInt32(values[0]);
            index2 = System.Convert.ToInt32(parameter);
            if (a == null)
                return null;
            return a[System.Convert.ToInt32(index1)][index2];
        }

        public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
        {
            a[index1][index2] = System.Convert.ToDouble(value);

            return new object[] { index1, a };
        }


        #endregion
    }
}
