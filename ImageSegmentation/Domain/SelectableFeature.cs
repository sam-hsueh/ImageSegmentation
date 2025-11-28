using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static SamSharp.Utils.Classes;

namespace ImageSegmentation.Domain
{
    public class SelectableFeature : ViewModelBase
    {
        private bool _isSelected;
        private int? _cat;
        private string? _description;
        private int _shape;
        private List<Point> _fpoints;//特征点
        private List<SamPoint> _cpoints;//sam点
        private byte[] _mask;
        public bool IsSelected
        {
            get => _isSelected;
            set => SetProperty(ref _isSelected, value);
        }

        public int Shape
        {
            get => _shape;
            set => SetProperty(ref _shape, value);
        }

        public int? Cat
        {
            get => _cat;
            set => SetProperty(ref _cat, value);
        }

        public string? Description
        {
            get => _description;
            set => SetProperty(ref _description, value);
        }

        public List<Point> FPoints
        {
            get => _fpoints;
            set => SetProperty(ref _fpoints, value);
        }
        public List<SamPoint> CPoints
        {
            get => _cpoints;
            set => SetProperty(ref _cpoints, value);
        }
        public byte[] Mask
        {
            get => _mask;
            set => SetProperty(ref _mask, value);
        }

    }
}
