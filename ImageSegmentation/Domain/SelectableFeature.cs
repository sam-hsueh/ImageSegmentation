using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImageSegmentation.Domain
{
    public class SelectableFeature : ViewModelBase
    {
        private bool _isSelected;
        private int? _cat;
        private string? _description;
        private int _shape;
        private List<Point> _fpoints;

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
    }
}
