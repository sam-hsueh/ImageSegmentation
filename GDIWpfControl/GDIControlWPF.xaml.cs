// Test control fronend for WPF for GDIPainter library
//   (c) Mokrov Ivan
// special for habrahabr.ru
// under MIT license
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms.Integration;
using System.Windows.Forms;

namespace GDIWpfControl
{
	public partial class GDIControlWPF : WindowsFormsHost
	{	
		public event EventHandler<MouseEventArgs> GdiMouseDownChanged;
        public event EventHandler<MouseEventArgs> GdiMouseMoveChanged;
        public event EventHandler<MouseEventArgs> GdiMouseUpChanged;
        public event EventHandler<EventArgs> GdiMouseLeaveChanged;
        public delegate void ContextMS(int e);
        public event ContextMS GdiContextDraw;


        private readonly HandleRef hDCRef;
		private readonly System.Drawing.Graphics hDCGraphics;
		private readonly GDIPainter RP;

        /// <summary>
        /// root Bitmap
        /// </summary>
        public System.Drawing.Bitmap BMP { get; private set; }

		/// <summary>
		/// Graphics object to paint on BMP
		/// </summary>
		public System.Drawing.Graphics GFX { get; private set; }
        public int[] _pArray => RP._pArray;

        /// <summary>
        /// Real per-pixel width of backend Win32 control, w/o DPI resizes of WPF layout
        /// </summary>
        public int Width { get { return GDIPaint.Width; } }
		/// <summary>
		/// Real per-pixel height of backend Win32 control, w/o DPI resizes of WPF layout
		/// </summary>
		public int Height { get { return GDIPaint.Height; } }

		/// <summary>
		/// Lock it to avoid resize/repaint race
		/// </summary>
		public readonly object Lock = new object();

		public GDIControlWPF()
		{
			InitializeComponent();
			hDCGraphics = GDIPaint.CreateGraphics();
			hDCRef = new HandleRef(hDCGraphics, hDCGraphics.GetHdc());
			RP = new GDIPainter();

			BMP = new System.Drawing.Bitmap(Width, Height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
			GFX = System.Drawing.Graphics.FromImage(BMP);

			GDIPaint.Resize += (sender, args) =>
			{
				lock (Lock)
				{
					if (GFX != null) GFX.Dispose();
					if (BMP != null) BMP.Dispose();
					BMP = new System.Drawing.Bitmap(Width, Height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
					GFX = System.Drawing.Graphics.FromImage(BMP);
				}
			};
		}
		/// <summary>
		/// After all in-memory paint on GFX, call it to display it on control
		/// </summary>
		public void Paint()
		{
			RP.Paint(hDCRef, BMP);
			IntPtr hDst = GFX.GetHdc();
			GFX.ReleaseHdc(hDst);
		}

		protected override void Dispose(bool disposing)
		{
			lock (this)
			{
				if (GFX != null) GFX.Dispose();
				if (BMP != null) BMP.Dispose();
				if (hDCGraphics != null) hDCGraphics.Dispose();
				RP.Dispose();
			}

			base.Dispose(disposing);
		}

        private void GDIPaint_MouseDown(object sender, MouseEventArgs e)
        {
			if(GdiMouseDownChanged!=null)
				GdiMouseDownChanged(this, e);
        }

        private void GDIPaint_MouseMove(object sender, MouseEventArgs e)
        {
			if(GdiMouseMoveChanged!=null) GdiMouseMoveChanged(this,e);
        }

        private void GDIPaint_MouseUp(object sender, MouseEventArgs e)
        {
			if(GdiMouseUpChanged!=null)
				GdiMouseUpChanged(this, e);
        }

        private void GDIPaint_MouseLeave(object sender, EventArgs e)
        {
			if (GdiMouseLeaveChanged != null) 
				GdiMouseLeaveChanged(this, e);

        }

        private void GDIPaint_ContextDraw(int e)
        {
            GdiContextDraw?.Invoke(e);
        }
    }
}
