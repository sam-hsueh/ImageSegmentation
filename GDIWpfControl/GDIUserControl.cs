using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace GDIWpfControl
{
    public partial class GDIUserControl : UserControl
    {
        public delegate void ContextMS(int e);
        public event ContextMS ContextDraw;
        public GDIUserControl()
        {
            InitializeComponent();
            //this.MouseDown += GDIUserControl_MouseDown;
            SetStyle(ControlStyles.DoubleBuffer, false);
            SetStyle(ControlStyles.UserPaint, true);
            SetStyle(ControlStyles.AllPaintingInWmPaint, true);
            SetStyle(ControlStyles.Opaque, true);

            this.MinimumSize = new System.Drawing.Size(100, 100);
            LoadMenustrip(this);
        }
        void LoadMenustrip(Control Control)
        {
            ContextMenuStrip ms = new ContextMenuStrip();
            ms.Items.Add("取消绘制(Esc)");
            ms.Items.Add("取消选择");
            ms.Items.Add("删除选框");
            ms.ItemClicked += new ToolStripItemClickedEventHandler(ms_ItemClicked);
            Control.ContextMenuStrip = ms;
        }
        void ms_ItemClicked(object sender, ToolStripItemClickedEventArgs e)
        {
            int r = -1;
            if(e.ClickedItem.Text== "取消绘制(Esc)")
            {
                r = 0;
            }
            else if(e.ClickedItem.Text == "取消选择")
            {
                r = 1;
            }
            else if(e.ClickedItem.Text == "删除选框")
            {
                r = 2;
            }
            if (ContextDraw != null)
            {
                ContextDraw(r);
            }            
        }
        //private void GDIUserControl_MouseDown(object sender, MouseEventArgs e)
        //{
        //    if (MoveDownChanged != null)
        //    {
        //        MoveDownChanged(this, e);
        //    }
        //}
    }
}
