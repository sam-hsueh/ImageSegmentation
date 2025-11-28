using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Forms;
using System.Windows.Input;
using ImageSegmentation.Domain;
using static SamSharp.Utils.Classes;
using Color = System.Drawing.Color;
using ColorConverter = System.Windows.Media.ColorConverter;
using ColorW = System.Windows.Media.Color;
using MessageBox = System.Windows.Forms.MessageBox;
using Pen = System.Drawing.Pen;
using Point = System.Drawing.Point;
using Window = System.Windows.Window;
namespace ImageSegmentation
{
    /// <summary>
    /// Interaction logic for MainWindoRWidth.xaml
    /// </summary>
    public partial class Home : System.Windows.Controls.UserControl
    {
        MainWindowViewModel? mwv;
        string curDir = "";
        Bitmap sourceBitmap, OrignalBitmap;
        private string modelPath;
        private SamSharp.Utils.SamPredictor predictor;
        bool MousePress = false;
        Pen gPen = new Pen(Color.Maroon, 1);
        Pen rPen = new Pen(Color.Red, 2);
        public Home()
        {
            InitializeComponent();
            LoadModel();
            Loaded += Home_Loaded;
            curDir = Properties.Settings.Default.InitDir;
            MaxF = Properties.Settings.Default.MaxF;
        }
        public void LoadModel()
        {
            if (Properties.Settings.Default.ModelIndex == 0)
                modelPath = AppDomain.CurrentDomain.BaseDirectory + @"\SamModels\mobile_sam.pt";
            else
            modelPath = AppDomain.CurrentDomain.BaseDirectory + @"\SamMpdels\sam_vit_b_01ec64.pth";
            var samscalarType = "BFloat16";
            if (Properties.Settings.Default.ScalarIndex == 0)
                samscalarType = "BFloat16";
            else
                samscalarType = "Float32";

            SamDevice device = (SamDevice)Enum.Parse(typeof(SamDevice), "CUDA");  // SamDevice.Cuda or SamDevice.CPU
            SamScalarType dtype = (SamScalarType)Enum.Parse(typeof(SamScalarType), samscalarType); // Float, Half or BF16.
            try
            {
                if (predictor == null)
                {
                    predictor = new SamSharp.Utils.SamPredictor(modelPath, device, dtype);
                    //MessageBox.Show("Model Loaded Done.");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }
        Color GetAlphaColor(Color color,int alpha)
        {
            return Color.FromArgb(alpha, color);
        }
        private void Home_Loaded(object sender, RoutedEventArgs e)
        {
            Window? window = Window.GetWindow(this);
            mwv = this.DataContext as MainWindowViewModel;
            mwv!.mainW = this;
            curDir = Properties.Settings.Default.InitDir;
            DataGrid_SelectionChanged(null, null);
            var timer = new System.Windows.Forms.Timer();
            timer.Interval = 50;
            timer.Tick += new System.EventHandler(timer_Tick);
            timer.Start();
        }
        public int ctime = 5;
        private void timer_Tick(object? sender, EventArgs e)
        {
            if (ctime > 0)
                DrawF();
            ctime--;
            if (ctime < 0)
                ctime = 0;
        }

        private void Home_Closing(object? sender, System.ComponentModel.CancelEventArgs e)
        {
            Properties.Settings.Default.Save();
            GC.Collect();
            System.Windows.Application.Current.Shutdown();
            Environment.Exit(0);
        }


        public static ColorW[]? colors { get; set; } = new ColorW[] { (ColorW)ColorConverter.ConvertFromString("#FF3F51B5"), (ColorW)ColorConverter.ConvertFromString("#FF3A7E00"), (ColorW)ColorConverter.ConvertFromString("#FFB00020") };

        public static ObservableCollection<SelectableFiles> FileList
        {
            set;
            get;
        } = new ObservableCollection<SelectableFiles>();
        ObservableCollection<string> CatList
        {
            set;
            get;
        } = new ObservableCollection<string>();


        private void Button_Click(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.FolderBrowserDialog openolderDialog = new FolderBrowserDialog();
            if (curDir == "")
                curDir = Environment.GetFolderPath(Environment.SpecialFolder.Personal);
            openolderDialog.InitialDirectory = curDir;
            var result = openolderDialog.ShowDialog();
            if (result == System.Windows.Forms.DialogResult.OK)
            {
                sourceBitmap = null;
                curDir = openolderDialog.SelectedPath;
                OpenFolder();
            }
            else
                return;
        }

        void OpenFolder()
        {
            string path = curDir;
            try
            {
                FileList.Clear();
                foreach (string Path in Directory.GetFiles(path))
                {
                    var PExt = new FileInfo(Path).Extension.ToLower();
                    if (PExt == ".jpg" || PExt == ".bmp" || PExt == ".png") //筛选图片格式
                    {
                        var fi = new FileInfo(Path);
                        FileList.Add(
                            new SelectableFiles() { FileName = fi.Name, Directory = fi.Directory.FullName }
                            );
                    }
                }
                mwv!.FileList = FileList;
                if (FileList.Count > 0)
                {
                    FileListGrid.SelectedIndex = 0;
                    //FileName = (string)FileListGrid.Items[FileListGrid.SelectedIndex];
                    for (int i = 0; i < FileList.Count; i++)
                    {
                        var imgFile = curDir + @"\Project\DataSets\Images\Train\"+ FileList[i].FileName;
                        var labelFile = curDir + @"\Project\DataSets\Labels\Train\" + FileList[i].FileName.Replace(new FileInfo(FileList[i].FileName).Extension.ToLower(), ".txt");
                        if (File.Exists(imgFile) && File.Exists(labelFile))
                        {
                            mwv!.FileList[i].IsSelected = true;
                        }
                    }
                }
                mwv!.OriginalImageDir = curDir;
                mwv!.ProjectDir = curDir + @"\Project";
            }
            catch(Exception ex) { throw ex; };
        }
        string FileName = "";
        bool drawPoly = false;
        bool hovarSP = false;
        List<Point> curPS;
        bool close = false;
        int[,] Map = null;
        static object _object = new object();
        float labelw = 22f;
        StringFormat sf = new StringFormat();
        float fontsize = 12f;
        float fontsize2 = 22f;
        int margin = 40;
        float rate = 15f;
        float wratio = 1.0f, hratio = 1.0f;
        public void DrawF(int x = 0, int y = 0)
        {
            if (sourceBitmap == null)
                return;
            lock (_object)
            {
                Bitmap bm = new Bitmap(displayBitmap);
                using (Graphics g = Graphics.FromImage(bm))
                {
                    if (drawPoly && x != 0 && y != 0)
                    {
                        if (curPS.Count == 0)
                        {
                            drawPoly = false;
                            return;
                        }
                        if (Math.Abs(x - curPS[0].X) < 15 && Math.Abs(y - curPS[0].Y) < 15 && curPS.Count > 2 && closed == false)
                        {
                            x = curPS[0].X;
                            y = curPS[0].Y;
                            g.FillEllipse(new SolidBrush(Color.Yellow), new Rectangle(x - 15, y - 15, 30, 30));
                            g.DrawEllipse(new Pen(Color.Red, 1), new Rectangle(x - 15, y - 15, 30, 30));
                            close = true;
                        }
                        else
                            close = false;
                        if (closed)
                        {
                            closed = false;
                            drawPoly = false;
                            curPS = null;
                        }
                        else
                        {
                            var cps = new Point[curPS.Count + 1];
                            Array.Copy(curPS.ToArray(), cps, curPS.Count);
                            cps[curPS.Count] = new Point(x, y);
                            g.DrawLines(rPen, cps);
                            for (int j = 0; j < cps.Length; j++)
                                g.FillRectangle(new SolidBrush(Color.DarkRed), new RectangleF(cps[j].X - 4, cps[j].Y - 4, 8, 8));
                        }
                    }
                    for (int i = 0; i < mwv!.FeatureList.Count; i++)
                    {
                        var s = mwv!.FeatureList[i].FPoints;
                        int cat = (int)mwv!.FeatureList[i].Cat;
                        Color c = PrimaryColor[cat%20];
                        if(mwv!.FeatureList[i].FPoints.Count>2)
                            g.DrawPolygon(new Pen(c, 3), mwv!.FeatureList[i].FPoints.ToArray());
                        for (int j = 0; j < mwv!.FeatureList[i].FPoints.Count; j++)
                            g.FillRectangle(new SolidBrush(c), new RectangleF(mwv!.FeatureList[i].FPoints[j].X - 4, mwv!.FeatureList[i].FPoints[j].Y - 4, 8, 8));
                        //矩形框中心点
                        if (mwv!.FeatureList[i].Shape == 0)
                            g.FillRectangle(new SolidBrush(c), new RectangleF((mwv!.FeatureList[i].FPoints[0].X + mwv!.FeatureList[i].FPoints[1].X) / 2 - 4, (mwv!.FeatureList[i].FPoints[0].Y + mwv!.FeatureList[i].FPoints[3].Y) / 2 - 4, 8, 8));
                        if (SelectedF == i&& mwv!.FeatureList[SelectedF].FPoints.Count>2)
                        {
                            g.FillPolygon(new SolidBrush(GetAlphaColor(c, 100)), mwv!.FeatureList[SelectedF].FPoints.ToArray());
                        }
                        if (HoverF == i)
                        {
                            if (HoverF != SelectedF)
                                g.FillPolygon(new SolidBrush(GetAlphaColor(c, 100)), mwv!.FeatureList[HoverF].FPoints.ToArray());
                            if (HoverFP >= 0)
                            {
                                g.FillRectangle(new SolidBrush(Color.Yellow), new RectangleF(mwv!.FeatureList[HoverF].FPoints[HoverFP].X - 6, mwv!.FeatureList[HoverF].FPoints[HoverFP].Y - 6, 12, 12));
                            }
                        }
                    }
                }
                DrawToGraphics(bm);
            }
            GC.Collect();
        }
        private unsafe void DrawToGraphics(Bitmap bm)
        {
            lock (GWpf.Lock)
            {
                GWpf.GFX.SmoothingMode = SmoothingMode.AntiAlias;
                GWpf.GFX.SmoothingMode = SmoothingMode.HighQuality;
                GWpf.GFX.CompositingQuality = CompositingQuality.HighQuality;
                GWpf.GFX.PixelOffsetMode = PixelOffsetMode.HighQuality;
                Graphics g = GWpf.GFX;
                if (bm == null)
                    return;
                g.Clear(Color.White);
                g.DrawImage(bm, 0, 0);
                GWpf.Paint();
            }
        }

        private void RWidth_ValueChanged(object sender, EventArgs e)
        {
            if (sourceBitmap == null || mwv!.FeatureList == null || mwv!.FeatureList.Count == 0 || SelectedF == -1)
                return;
            var cur = mwv!.FeatureList[SelectedF];
            DrawF();
            DrawMap();
        }

        StringBuilder sb = new StringBuilder();
        private unsafe void DrawMap()
        {
            if (sourceBitmap == null)
                return;
            //sb.Clear();
            lock (_object)
            {
                Bitmap bm = new Bitmap(sourceBitmap.Width, sourceBitmap.Height);
                using (Graphics g = Graphics.FromImage(bm))
                {
                    g.SmoothingMode = SmoothingMode.HighQuality;
                    g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    g.Clear(Color.Black);
                    for (int i = 0; i < mwv!.FeatureList.Count; i++)
                    {
                        var s = mwv!.FeatureList[i].FPoints.ToArray();
                        if (s.Length < 3)
                            continue;
                        int cv = (i + 1) * 10;
                        var cl = Color.FromArgb(255, cv, cv, cv);
                        g.FillPolygon(new SolidBrush(cl), s);
                        g.DrawPolygon(new Pen(cl,2), s);
                    }
                }
                int height = bm.Height;
                int width = bm.Width;
                BitmapData sourceData = bm.LockBits(new Rectangle(0, 0,
                                         width, height),
                                                           ImageLockMode.ReadOnly,
                                                     bm.PixelFormat);
                byte* src = (byte*)sourceData.Scan0.ToPointer();
                int stride = sourceData.Stride;

                int channels = System.Drawing.Image.GetPixelFormatSize(bm.PixelFormat) / 8;
                int offset = stride - width * channels;
                int r = stride;
                int r2 = 2 * stride;
                int r3 = 3 * stride;
                int c = channels;
                int c2 = 2 * channels;
                int c3 = 3 * channels;
                int[] v = new int[33];
                Map = new int[height, width];
                try
                {
                    unsafe
                    {
                        fixed (int* dstPtr = Map)
                        {
                            int* dst = dstPtr;
                            for (int i = 0; i < height; i++)
                            {
                                for (int j = 0; j < width; j++, src += channels, dst++)
                                {
                                    byte* s = src;
                                    if ((s[0] > 0 || s[1] > 0 || s[2] > 0)&&(i>=3&&i<height-3&&j>=3&&j<width-3))
                                    {
                                        //int cp=
                                        v[0] = s[-c3 - r3 + 1];
                                        v[1] = s[-c2 - r3 + 1];
                                        v[2] = s[-c - r3 + 1];
                                        v[3] = s[-r3 + 1];
                                        v[4] = s[c - r3 + 1];
                                        v[5] = s[c2 - r3 + 1];
                                        v[6] = s[c2 - r3 + 1];

                                        v[7] = s[-c3 - r2 + 1];
                                        v[8] = s[-c3 - r + 1];
                                        v[9] = s[-c3 + 1];
                                        v[10] = s[-c3 + r + 1];
                                        v[11] = s[-c3 + r2 + 1];

                                        v[12] = s[-r2 + 1];
                                        v[13] = s[-r + 1];
                                        v[14] = s[1];
                                        v[15] = s[r + 1];
                                        v[16] = s[r2 + 1];

                                        v[17] = s[-c2 + 1];
                                        v[18] = s[-c + 1];
                                        v[19] = s[c + 1];
                                        v[20] = s[c2 + 1];

                                        v[21] = s[c3 - r2 + 1];
                                        v[22] = s[c3 - r + 1];
                                        v[23] = s[c3 + 1];
                                        v[24] = s[c3 + r + 1];
                                        v[25] = s[c3 + r2 + 1];
                                        v[26] = s[-c3 + r3 + 1];
                                        v[27] = s[-c2 + r3 + 1];
                                        v[28] = s[-c + r3 + 1];
                                        v[29] = s[r3 + 1];
                                        v[30] = s[c + r3 + 1];
                                        v[31] = s[c2 + r3 + 1];
                                        v[32] = s[c3 + r3 + 1];
                                        *dst = v.Max() / 10;
                                    }
                                }
                                src += offset;
                            }
                        }
                    }
                    bm.UnlockBits(sourceData);
                }
                catch { }
            }
            GC.Collect();
        }
        unsafe int GetPolyIndex(int x, int y)
        {
            if (Map == null || x <= 0 || y <= 0 || x >= sourceBitmap.Width - 1 || y >= sourceBitmap.Height - 1)
                return 0;
            //return Map[y,x];

            var m = Math.Max(Math.Max(Map[y - 1, x], Map[y + 1, x]), Math.Max(Map[y, x + 1], Map[y, x - 1]));
            return Math.Max(Map[y, x], m);
        }

        private void DataGrid_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (FileListGrid.SelectedIndex > -1)
            {
                try
                {
                    int k = FileListGrid.SelectedIndex;
                    FileName = mwv!.FileList[k].FileName;
                    string FilePath = curDir + @"\" + FileName;
                    StreamReader streamReader = new StreamReader(FilePath);
                    OrignalBitmap = (Bitmap)Bitmap.FromStream(streamReader.BaseStream);
                    streamReader.Close();

                    var image = Cv2.ImRead(FilePath);
                    predictor.SetImage(image, 1024);

                    int width = (int)GWpf.Width;
                    int height = (int)GWpf.Height;
                    if (width < 0)
                        return;
                    wratio = 1.0f;
                    hratio = 1.0f;
                    //if (SketchImg.IsChecked == true)
                    {
                        wratio = (float)((float)OrignalBitmap.Width / (float)width);
                        hratio = (float)((float)OrignalBitmap.Height / (float)height);
                        sourceBitmap = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format32bppRgb);
                        sourceBitmap.SetResolution(OrignalBitmap.HorizontalResolution, OrignalBitmap.VerticalResolution);
                        Graphics graphic = Graphics.FromImage(sourceBitmap);
                        graphic.SmoothingMode = SmoothingMode.HighQuality;
                        graphic.InterpolationMode = InterpolationMode.HighQualityBicubic;
                        graphic.DrawImage(OrignalBitmap, new Rectangle(0, 0, width, height));
                        graphic.Dispose();
                    }
                    displayBitmap = BitmapAdjust(sourceBitmap, (float)Brightness.Value / 100f, (float)(Contrast.Value) / 100f);
                    richtextBox1.Document.Blocks.Clear();
                    richtextBox1.AppendText("原图：" + OrignalBitmap.Width + " X " + OrignalBitmap.Height + " X " + System.Drawing.Image.GetPixelFormatSize(OrignalBitmap.PixelFormat) / 8);
                    richtextBox1.AppendText("\r\n现图：" + sourceBitmap.Width + " X " + sourceBitmap.Height + " X " + System.Drawing.Image.GetPixelFormatSize(sourceBitmap.PixelFormat) / 8);
                    //DrawToGraphics(displayBitmap);
                    if (mwv!.FeatureList.Count > 0)
                    {
                        bool isOut = false;
                        for (int i = 0; i < mwv!.FeatureList.Count; i++)
                        {
                            for (int j = 0; j < mwv!.FeatureList[i].FPoints.Count; j++)
                            {
                                if (mwv!.FeatureList[i].FPoints[j].X > sourceBitmap.Width - 6 || mwv!.FeatureList[i].FPoints[j].Y > sourceBitmap.Height - 6)
                                {
                                    isOut = true;
                                }
                            }
                        }
                        if (isOut)
                        {
                            mwv!.FeatureList.Clear();
                        }
                    }
                    var labelFile = curDir + @"\Project\DataSets\Labels\Train\" + FileList[k].FileName.Replace(new FileInfo(FileList[k].FileName).Extension.ToLower(), ".txt");
                    if (File.Exists(labelFile))
                    {
                        mwv!.FeatureList.Clear();
                        using (StreamReader sr = File.OpenText(labelFile))
                        {
                            string s = "";
                            while ((s = sr.ReadLine()) != null)
                            {
                                string[] arr = s.Split(' ');
                                if(arr.Length==9)
                                {
                                    int cat = int.Parse(arr[0]);
                                    int p1x = (int)(double.Parse(arr[1]) * OrignalBitmap.Width / wratio);
                                    int p1y = (int)(double.Parse(arr[2]) * OrignalBitmap.Height / hratio);
                                    int p2x = (int)(double.Parse(arr[3]) * OrignalBitmap.Width / wratio);
                                    int p2y = (int)(double.Parse(arr[4]) * OrignalBitmap.Height / hratio);
                                    int p3x = (int)(double.Parse(arr[5]) * OrignalBitmap.Width / wratio);
                                    int p3y = (int)(double.Parse(arr[6]) * OrignalBitmap.Height / hratio);
                                    int p4x = (int)(double.Parse(arr[7]) * OrignalBitmap.Width / wratio);
                                    int p4y = (int)(double.Parse(arr[8]) * OrignalBitmap.Height / hratio);

                                    var ps = new Point[] { new Point(p1x, p1y), new Point(p2x, p2y), new Point(p3x, p3y), new Point(p4x, p4y) };
                                    mwv!.FeatureList.Add(new SelectableFeature() { FPoints = ps.ToList(), Shape = 0, Cat = cat, Description = "Rectangle" });
                                }
                                else
                                {
                                    int cat = int.Parse(arr[0]);
                                    Point[] ps = new Point[(arr.Length - 1)/2];
                                    int n = 0;
                                    for(int m = 1;m < arr.Length - 1;m+=2)
                                    {
                                        int x = (int)(double.Parse(arr[m]) * OrignalBitmap.Width / wratio);
                                        int y = (int)(double.Parse(arr[m + 1]) * OrignalBitmap.Height / hratio);
                                        ps[n++] = new Point(x, y);
                                    }
                                    mwv!.FeatureList.Add(new SelectableFeature() { FPoints = ps.ToList(), Shape = 1, Cat = cat, Description = "Polygon" });
                                }
                            }
                     //       FeaturesList.SelectedIndex = 0;
                        }
                    }
                    DrawF();
                    DrawMap();
                }
                catch(Exception ex )
                {
                }
            }
        }
        void SaveFeatureFile()
        {
            if (sourceBitmap == null || mwv!.FeatureList.Count == 0 || FileName == "")
                return;
            string ProjectDataPath = (curDir + @"\Project\DataSets");
            string imgPath = ProjectDataPath + @"\Images\Train\";
            string labelPath = ProjectDataPath + @"\Labels\Train\";
            if (!Directory.Exists(imgPath))
                Directory.CreateDirectory(imgPath);
            if (!Directory.Exists(labelPath))
                Directory.CreateDirectory(labelPath);
            var oimgPath = curDir + @"\" + FileName;
            var nimgPath = imgPath + @"\" + FileName;
            File.Copy(oimgPath, nimgPath, true);
            var lFile =  labelPath + @"\" + FileName.Replace(new FileInfo(FileName).Extension.ToLower(), ".txt");
            sb = new StringBuilder();
            {
                for (int i = 0; i < mwv!.FeatureList.Count; i++)
                {
                    sb.Append(mwv!.FeatureList[i].Cat + " ");
                    for (int j = 0; j < mwv!.FeatureList[i].FPoints.Count; j++)
                    {
                        sb.Append((mwv!.FeatureList[i].FPoints[j].X * wratio / OrignalBitmap.Width) + " " + (mwv!.FeatureList[i].FPoints[j].Y * hratio / OrignalBitmap.Height));
                        if (j != mwv!.FeatureList[i].FPoints.Count - 1)
                            sb.Append(" ");
                    }
                    if (i != mwv!.FeatureList.Count - 1)
                        sb.Append("\n");
                }
                using (TextWriter textWriter = new StreamWriter(new FileStream(lFile, FileMode.Create, FileAccess.Write, FileShare.Write, 4096, true), System.Text.Encoding.Default))
                {
                    textWriter.Write(sb.ToString());
                }
                mwv!.FileList[FileListGrid.SelectedIndex].IsSelected = true;
            }
        }
        public string ImageToBase64(string imgpath)
        {
            using (System.Drawing.Image image = System.Drawing.Image.FromFile(imgpath))
            {
                using (MemoryStream m = new MemoryStream())
                {
                    image.Save(m, image.RawFormat);
                    byte[] imageBytes = m.ToArray();
                    var base64String = Convert.ToBase64String(imageBytes);
                    return base64String;
                }
            }
        }
        static string ConvertImageToBase64(string imagePath)
        {
            byte[] imageBytes = File.ReadAllBytes(imagePath);
            return Convert.ToBase64String(imageBytes);
        }
        public static string ToPixelBuffer2(string FilePath)
        {
            Bitmap bmp = (Bitmap)Bitmap.FromFile(FilePath);
            BitmapData sourceData =
                       bmp.LockBits(new Rectangle(0, 0,
                       bmp.Width, bmp.Height),
                       ImageLockMode.ReadOnly,
                       System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                       //bmp.PixelFormat);
            byte[] pixelBuffer = new byte[sourceData.Stride *
                                          sourceData.Height];

            Marshal.Copy(sourceData.Scan0, pixelBuffer, 0,
                                       pixelBuffer.Length);

            bmp.UnlockBits(sourceData);
            return Convert.ToBase64String(pixelBuffer);
        }

        public static byte[] ToPixelBuffer(string FilePath)
        {
            Bitmap bmp = (Bitmap)Bitmap.FromFile(FilePath);
            BitmapData sourceData =
                       bmp.LockBits(new Rectangle(0, 0,
                       bmp.Width, bmp.Height),
                       ImageLockMode.ReadOnly,
                       bmp.PixelFormat);
            int Channels = System.Drawing.Image.GetPixelFormatSize(bmp.PixelFormat) / 8;
            byte[] pixelBuffer = new byte[sourceData.Width * Channels * sourceData.Height];
            unsafe
            {
                // base pointers
                byte* src = (byte*)sourceData.Scan0.ToPointer();
                fixed (byte* dst = pixelBuffer)
                {
                    int sourceStride = sourceData.Stride;
                    for (int y = 0; y < sourceData.Height; y++)
                    {
                        byte* s = (byte*)(src + y * sourceStride);
                        byte* d = (byte*)(dst + y * sourceData.Width * Channels);

                        for (int x = 0; x < sourceData.Width; x++, s += Channels, d += Channels)
                        {
                            for (int i = 0; i < Channels; i++)
                            {
                                d[i] = s[i];
                            }
                        }
                    }
                }
            }
            bmp.UnlockBits(sourceData);
            return pixelBuffer;
            //return Convert.ToBase64String(pixelBuffer);
        }

        private void pictureBox1_SizeChanged(object sender, EventArgs e)
        {
            DataGrid_SelectionChanged(null, null);
        }

        private void FeaturesList_SelectedIndexChanged(object sender, EventArgs e)
        {
        }

        private void Form_KeyDown(object sender, System.Windows.Input.KeyEventArgs e)
        {
            if (e.Key == Key.Escape)
            {
                if (curPS != null && curPS.Count > 0)
                { curPS.Clear(); DrawF(); }
                else if (SelectedF >= 0 || HoverFP >= 0)
                {
                    SelectedF = -1;
                    HoverFP = -1;
                    DrawF();
                }
            }
            else if (e.Key == Key.Delete)
            {
                if (mwv!.FeatureList != null && mwv!.FeatureList.Count > 0)
                {
                    mwv!.FeatureList.RemoveAt(SelectedF);
                    DrawF();
                    DrawMap();
                }
            }
        }
        int MaxF = Properties.Settings.Default.MaxF;
        private List<SamPoint> samPoints = new List<SamPoint>();
        private void GWpf_MouseDown(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            if (sourceBitmap == null || e.Button == MouseButtons.Right)
                return;
            if (e.X >= 5 && e.X < sourceBitmap.Width - 5 && e.Y >= 5 && e.Y < sourceBitmap.Height - 5)
            {
                if (HoverF >= 0)
                {
                    SelectedF = HoverF;
                }
                else
                {
                    if (SAMPolygon.IsChecked == true)
                    {
                        samPoints.Clear();
                        int rr = e.X;
                        int sr = e.Location.X;
                        SamPoint point = new SamPoint((int)(e.X*wratio), (int)(e.Y*hratio), true); // Assuming true for foreground
                        samPoints.Add(point);

                        List<PredictOutput> outputs = predictor.Predict(samPoints);
                        Thread.Sleep(100);
                        PredictOutput output = outputs[0];

                        bool[,] mask = output.Mask;
                        int length = OrignalBitmap.Height * OrignalBitmap.Width; // or src.Height * src.Step;
                        byte[] mask1 = new byte[length];
                        Buffer.BlockCopy(mask, 0, mask1, 0, length);
                        var ms = CheckMaskCrossAndMerge(ref mask1);                        
                        var src = new Mat(OrignalBitmap.Width, OrignalBitmap.Height, MatType.CV_8UC1);
                        Marshal.Copy(mask1, 0, src.Data, length);
                        OpenCvSharp.Point[][] contours;
                        Cv2.FindContours(
                            image: src,
                            contours: out contours,
                            hierarchy: out HierarchyIndex[] outputArray,
                            mode: (RetrievalModes)RetrievalModes.External,
                            method: (ContourApproximationModes)ContourApproximationModes.ApproxSimple
                            );
                        List<OpenCvSharp.Point[]> query = contours.ToList<OpenCvSharp.Point[]>().OrderByDescending(t => Cv2.ContourArea(t)).Select(t => t).Take(1).ToList();
                        if (query.Count == 0)
                            return;
                        var epsilon = Cv2.ArcLength(query[0], true) * 0.004;
                        var approxContour = Cv2.ApproxPolyDP(query[0], epsilon, true);
                        curPS = new List<Point>();
                        for (int j = 0; j < approxContour.Length; j++)
                        {
                            curPS.Add(new Point((int)(approxContour[j].Y/wratio), (int)(approxContour[j].X/hratio)));
                        }
                        var nps =new Point[curPS.Count];
                        curPS.CopyTo(nps);
                        List<SamPoint> sp = new List<SamPoint>();
                        for (int i = 0; i < samPoints.Count; i++)
                        {
                            sp.Add(new SamPoint(samPoints[i].X, samPoints[i].Y, true));
                        }
                        if(ms==null)
                            mwv!.FeatureList.Add(new SelectableFeature() { FPoints = nps.ToList(), Shape = 1, Cat = 2, CPoints = sp, Mask = mask1, Description = "Polygon" });
                        else
                        {
                            int index = ms.Min();
                            ms = ms.OrderByDescending(t => t).ToList();
                            for (int i = 0; i < ms.Count; i++)
                            {
                                if (index != ms[i])
                                    mwv!.FeatureList.RemoveAt(ms[i]);
                            }
                            mwv!.FeatureList[index].FPoints = nps.ToList();
                            mwv!.FeatureList[index].Mask = mask1;
                            SelectedF = index;
                        }
                        DrawMap();
                        DrawF();
                        curPS.Clear();
                    }
                }

                    if (SAMPolygon.IsChecked == false/*&&BRect.IsChecked == false */&& BPolygon.IsChecked == true && HoverF < 0)
                    {
                        if (curPS == null || curPS.Count == 0)
                        {
                            curPS = new List<Point>();
                            curPS.Add(new Point((int)e.X, (int)e.Y));
                            drawPoly = true;
                        }
                        else
                        {
                            if (close && mwv!.FeatureList.Count < MaxF)
                            {
                                mwv!.FeatureList.Add(new SelectableFeature() { FPoints = curPS, Shape = 1, Cat = 2, Description = "Polygon" });
                                DrawMap();
                                closed = true;
                            }
                            else
                                curPS.Add(new Point((int)e.X, (int)e.Y));
                        }
                    }
                    else
                    {
                        if (HoverF == -1)
                        {
                        }
                        else
                        {
                            if (SelectedF >= 0)
                            {
                                mousePx = (int)e.X;
                                mousePy = (int)e.Y;
                                MousePress = true;
                                FeaturesList.SelectedIndex = SelectedF;
                            }
                        }
                    }
            }
        }
        private void GWpf_MouseMove(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            if (sourceBitmap == null || /*BRect.IsChecked == true && (e.X < 5 + RWidth.Value / 2 && e.X >= sourceBitmap.Width - 5 - RWidth.Value / 2 && e.Y < 5 + RHeight.Value / 2 && e.Y >= sourceBitmap.Height - 5 - RHeight.Value / 2) ||BRect.IsChecked == false &&*/ (e.X < 5 && e.X >= sourceBitmap.Width - 5 && e.Y < 5 && e.Y >= sourceBitmap.Height - 5))
                return;
            if (drawPoly)
            {
                DrawF((int)e.X, (int)e.Y);
            }
            else
            {
                if (MousePress && (e.Button == MouseButtons.Left/*||e.Button==MouseButtons.Right*/))
                {
                    if (SelectedF >= 0 && HoverFP < 0&& SelectedF< mwv!.FeatureList.Count)
                    {
                        GWpf.Cursor = System.Windows.Input.Cursors.SizeAll;
                        var curF = mwv!.FeatureList[SelectedF];
                        var dx = e.X - mousePx;
                        var dy = e.Y - mousePy;
                        bool valid = false;
                        for (int i = 0; i < curF.FPoints.Count; i++)
                        {
                            Point s = curF.FPoints[i];
                            var sX = s.X + dx;
                            var sY = s.Y + dy;
                            if (sX < 5 || sY < 5 || sX >= sourceBitmap.Width - 5 || sY >= sourceBitmap.Height - 5)
                            {
                                valid = true;
                                break;
                            }
                        }
                        if (!valid)
                            for (int i = 0; i < curF.FPoints.Count; i++)
                            {
                                Point s = curF.FPoints[i];
                                s.X = s.X + (int)dx;
                                s.Y = s.Y + (int)dy;
                                curF.FPoints[i] = s;
                            }
                    }
                    else if (SelectedF >= 0 && HoverFP >= 0)
                    {
                        GWpf.Cursor = System.Windows.Input.Cursors.Hand;

                        var curF = mwv!.FeatureList[SelectedF];
                        var dx = e.X - mousePx;
                        var dy = e.Y - mousePy;
                        if (curF.Shape == 0)
                        {
                            Point c = curF.FPoints[HoverFP];
                            int lp = HoverFP == 0 ? 3 : HoverFP - 1;
                            Point l = curF.FPoints[lp];
                            int np = HoverFP == 3 ? 0 : HoverFP + 1;
                            Point n = curF.FPoints[np];
                            int np2 = HoverFP > 1 ? HoverFP - 2 : HoverFP + 2;
                            Point n2 = curF.FPoints[np2];

                            var cx = c.X + dx;
                            var cy = c.Y + dy;
                            if (cx < 5)
                                cx = 5;
                            if (cy < 5)
                                cy = 5;
                            if (cx >= sourceBitmap.Width - 5)
                                cx = sourceBitmap.Width - 6;
                            if (cy >= sourceBitmap.Height - 5)
                                cy = sourceBitmap.Height - 6;
                            if (l.X == c.X)
                            {
                                c.X = (int)cx;
                                l.X = (int)cx;
                                c.Y = (int)cy;
                                n.Y = (int)cy;
                            }
                            else
                            {
                                c.X = (int)cx;
                                n.X = (int)cx;
                                c.Y = (int)cy;
                                l.Y = (int)cy;
                            }
                            if (Math.Abs(c.X - n2.X) <= 15 || Math.Abs(c.Y - n2.Y) <= 15/*|| Math.Abs(c.X - n2.X)>RWidth.Maximum|| Math.Abs(c.Y - n2.Y) > RHeight.Value*/)
                                return;
                            curF.FPoints[HoverFP] = c;
                            curF.FPoints[lp] = l;
                            curF.FPoints[np] = n;
                        }
                        else
                        {
                            Point s = curF.FPoints[HoverFP];
                            s.X += (int)dx;
                            s.Y += (int)dy;
                            if (s.X < 5)
                                s.X = 5;
                            if (s.Y < 5)
                                s.Y = 5;
                            if (s.X >= sourceBitmap.Width - 5)
                                s.X = sourceBitmap.Width - 6;
                            if (s.Y >= sourceBitmap.Height - 5)
                                s.Y = sourceBitmap.Height - 6;
                            curF.FPoints[HoverFP] = s;
                        }
                    }
                    mousePx = (int)e.X;
                    mousePy = (int)e.Y;
                    DrawF();
                    DrawMap();
                }
                else
                {
                    GWpf.Cursor = System.Windows.Input.Cursors.None;

                    int rev = GetPolyIndex((int)e.X, (int)e.Y);
                    if (rev > 0)
                    {
                        HoverF = rev - 1;
                        GWpf.Cursor = System.Windows.Input.Cursors.SizeAll;
                    }
                    for (int i = 0; i < mwv!.FeatureList.Count; i++)
                    {
                        var s = mwv!.FeatureList[i];
                        for (int j = 0; j < s.FPoints.Count; j++)
                        {
                            if (Math.Abs(s.FPoints[j].X - e.X) < 8 && Math.Abs(s.FPoints[j].Y - e.Y) < 8)
                            {
                                HoverF = i;
                                HoverFP = j;
                                GWpf.Cursor = System.Windows.Input.Cursors.Hand;
                                DrawF();
                                return;
                            }
                        }
                    }
                    HoverF = rev - 1;
                    HoverFP = -1;
                    DrawF();
                }
            }
        }

        private void GWpf_MouseUp(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            if (drawPoly == true)
                return;
            MousePress = false;
            DrawF();
        }
        public static Bitmap BitmapAdjust(Bitmap bmp, float brightness, float contrast/*, float gamma*/)
        {
            Bitmap dst = new Bitmap(bmp.Width, bmp.Height, bmp.PixelFormat);

            // Create the ImageAttributes object and apply the ColorMatrix
            ImageAttributes attributes = new ImageAttributes();
            ColorMatrix matrix = new ColorMatrix(new float[][]{
                  new float[] {contrast, 0, 0, 0, 0}, // scale red
                  new float[] {0, contrast, 0, 0, 0}, // scale green
                  new float[] {0, 0, contrast, 0, 0}, // scale blue
                  new float[] {0, 0, 0, 1.0f, 0}, // don't scale alpha
                  new float[] {brightness, brightness, brightness, 0, 1}
            });

            using (Graphics g = Graphics.FromImage(dst))
            {
                attributes.SetColorMatrix(matrix, ColorMatrixFlag.Default, ColorAdjustType.Bitmap);
                //                attributes.SetGamma(gamma);
                g.DrawImage(bmp,
                            new Rectangle(0, 0, dst.Width, dst.Height),
                            0, 0, dst.Width, dst.Height,
                            GraphicsUnit.Pixel,
                            attributes);
            }
            return dst;
        }

        private void GWpf_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            DataGrid_SelectionChanged(null, null);
        }

        private void Contrast_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (displayBitmap == null)
                return;
            displayBitmap = BitmapAdjust(sourceBitmap, (float)Brightness.Value / 100f, (float)(Contrast.Value) / 100f);
            DrawF();
        }

        private void SketchImg_Checked(object sender, RoutedEventArgs e)
        {
            DataGrid_SelectionChanged(null, null);
        }

        private void MNext_Click(object sender, RoutedEventArgs e)
        {
            SaveFeatureFile();
            if (FileListGrid.SelectedIndex < FileListGrid.Items.Count - 1)
                FileListGrid.SelectedIndex++;
        }


        private void FeaturesList_SelectedIndexChanged(object sender, SelectionChangedEventArgs e)
        {
            if (mwv!.FeatureList.Count > 0 && FeaturesList.SelectedItems.Count > 0)
            {
                SelectedF = FeaturesList.SelectedIndex;
                BPolygon.IsChecked = true;
            }
            else
            {
                SelectedF = -1;
            }
            DrawF();
        }

        Bitmap displayBitmap;

        bool closed = false;
        int HoverF = -1, SelectedF = -1, HoverFP = -1/*, SelectedFP = -1*/;

        private void GWpf_GdiContextDraw(int e)
        {
            if (e == 0)
            {
                if (curPS != null && curPS.Count > 0)
                { curPS.Clear(); DrawF(); }
            }
            else if (e == 1)
            {
                SelectedF = -1;
            }
            else if (e == 2)
            {
                if (HoverF >= 0)
                {
                    if (mwv!.FeatureList != null && mwv!.FeatureList.Count > 0&& HoverF< mwv!.FeatureList.Count)
                    {
                        mwv!.FeatureList.RemoveAt(HoverF);
                        DrawF();
                        DrawMap();
                    }
                }
            }
            else if (e == 3)
            {
                if (HoverF >= 0&& HoverFP>=0)
                {
                    if(mwv!.FeatureList != null && mwv!.FeatureList[HoverF]!=null&& mwv!.FeatureList[HoverF].FPoints.Count>3 && mwv!.FeatureList[HoverF].FPoints[HoverFP]!=null)
                    {
                        mwv!.FeatureList[HoverF].FPoints.RemoveAt(HoverFP);
                        HoverFP = -1;
                        DrawF();
                        DrawMap();
                    }
                }
            }
        }

        int mousePx = -1, mousePy = -1;
        
        List<int> CheckMaskCrossAndMerge(ref byte[] Mask)
        {
            List<int> ms = new List<int>();
            byte[] mask = (byte[])Mask.Clone();
            for (int i = 0; i < mwv!.FeatureList.Count; i++)
            {
                if (mwv!.FeatureList[i].Mask!=null)
                {
                    for (int j = 0; j < mask.Length; j++) 
                    {
                        if (mask[j]==1&& mwv!.FeatureList[i].Mask[j]==1)
                        {
                            ms.Add(i); break;
                        }
                    }
                }
            }
            if (ms.Count == 0)
                return null;
            for (int i = 0; i < mask.Length; i++)
            {
                byte b = mask[i];
                if (b == 1)
                    continue;
                for (int j = 0; j < ms.Count; j++)
                {
                    if (mwv!.FeatureList[ms[j]].Mask[i] > 0)
                    {
                        b = 1;
                        break;
                    }
                }
                mask[i] = b;
            }
            Mask = mask;
            return ms;
        }
        private void ContextMenuFPoints_Click(object sender, RoutedEventArgs e)
        {
            if (curPS != null && curPS.Count > 0)
            { curPS.Clear(); DrawF(); }
        }

        private void ContextMenuCat_Click(object sender, RoutedEventArgs e)
        {
            if (SelectedF >= 0)
            {
                if (mwv!.FeatureList != null && mwv!.FeatureList.Count > 0 && SelectedF >= 0)
                {
                    mwv!.FeatureList.RemoveAt(SelectedF);
                    FeaturesList.Items.RemoveAt(SelectedF);
                    DrawF();
                    DrawMap();
                }
            }
        }

        private void ContextMenuSharp_Click(object sender, RoutedEventArgs e)
        {
            SelectedF = -1;
        }
        Color[] PrimaryColor => new Color[]
        {
            Color.Red,
            Color.Lime,
            Color.Purple,
            Color.Maroon,
            Color.DarkOrange,
            Color.Indigo,
            Color.Blue,
            Color.LightBlue,
            Color.LightGreen,
            Color.Cyan,
            Color.Teal,
            Color.Yellow,
            Color.Green,
            Color.Orange,
            Color.Brown,
            Color.Chocolate,
            Color.DarkSalmon,
            Color.Aquamarine,
            Color.DarkSeaGreen,
            Color.Pink,
      };
    }
}
