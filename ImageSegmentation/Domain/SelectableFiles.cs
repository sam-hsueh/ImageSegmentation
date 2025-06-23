using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImageSegmentation.Domain
{
    public class SelectableFiles : ViewModelBase
    {
        private bool _isSelected;
        private string? _filename;
        private string? _ditectory;
        private int? _index;
        public bool IsSelected
        {
            get => _isSelected;
            set => SetProperty(ref _isSelected, value);
        }


        public string? FileName
        {
            get => _filename;
            set => SetProperty(ref _filename, value);
        }

        public string? Directory
        {
            get => _ditectory;
            set => SetProperty(ref _ditectory, value);
        }

        public int? Index
        {
            get => _index;
            set => SetProperty(ref _index, value);
        }
    }
}
