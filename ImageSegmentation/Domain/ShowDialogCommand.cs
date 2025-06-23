using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace ImageSegmentation.Domain
{
    public class ShowDialogCommand : ICommand
    {
        public event EventHandler CanExecuteChanged;

        public Action _execute;
        public Func<bool> _canExecute;


        public ShowDialogCommand(Action execute)
        {
            _execute = execute;
        }


        public bool CanExecute(object parameter)
        {
            if (this._canExecute == null)
            {
                return true;
            }
            return _canExecute();
        }

        public void Execute(object parameter)
        {
            this._execute.Invoke();
        }
    }
}
