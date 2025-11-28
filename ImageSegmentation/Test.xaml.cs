using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Windows;
using TorchSharp.Modules;
using ImageSegmentation.Domain;
using YoloSharp.Models;
using YoloSharp.Types;
using Color = System.Drawing.Color;
using Point = OpenCvSharp.Point;
using Size = OpenCvSharp.Size;

namespace ImageSegmentation
{
    /// <summary>
    /// Interaction logic for MainWindoRWidth.xaml
    /// </summary>
    public partial class Test : System.Windows.Controls.UserControl
    {
        MainWindowViewModel? mwv;
        string curDir = "";
        public Test()
        {
            InitializeComponent();
            Loaded += Test_Loaded;
        }

        private void Test_Loaded(object sender, RoutedEventArgs e)
        {
            System.Windows.Window? window =  System.Windows.Window.GetWindow(this);
            mwv = this.DataContext as MainWindowViewModel;
            mwv!.testW = this;
            var timer = new System.Windows.Forms.Timer();
            timer.Interval = 250;
            timer.Tick += new System.EventHandler(timer_Tick);
            timer.Start();

        }
        public ObservableCollection<SelectableFiles> FileList = new ObservableCollection<SelectableFiles>();
        public ObservableCollection<SelectableFiles> TFileList = new ObservableCollection<SelectableFiles>();

        private unsafe void Button_Click(object sender, RoutedEventArgs e)
        {
            Microsoft.Win32.OpenFileDialog openFileDialog = new Microsoft.Win32.OpenFileDialog();
            curDir = mwv!.OriginalImageDir;
            if (curDir == "")
                curDir = Environment.GetFolderPath(Environment.SpecialFolder.Personal);
            openFileDialog.InitialDirectory = curDir;
            openFileDialog.Multiselect = true;
            bool result = openFileDialog.ShowDialog() ?? false;
            if (result)
            {
                try
                {
                    FileList.Clear();
                    foreach (string Path in openFileDialog.FileNames)
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
                    mwv!.TestFileList = FileList;
                    if (FileList.Count > 0)
                    {
                        var timgDir = mwv!.ProjectDir + @"\TestImages\";
                        if (!Directory.Exists(timgDir))
                            Directory.CreateDirectory(timgDir);
                        foreach (var f in Directory.GetFiles(timgDir))
                        {
                            File.Delete(f);
                        }
                        TestFileListGrid.SelectedIndex = 0;
                        for (int i = 0; i < FileList.Count; i++)
                        {
                            var oimgFile = FileList[i].Directory + @"\" + FileList[i].FileName;
                            var imgFile = mwv!.ProjectDir + @"\DataSets\Images\Train\" + FileList[i].FileName;
                            var labelFile = mwv!.ProjectDir + @"\DataSets\Labels\Train\" + FileList[i].FileName.Replace(new FileInfo(FileList[i].FileName).Extension.ToLower(), ".txt");
                            if (File.Exists(imgFile) && File.Exists(labelFile))
                            {
                                mwv.TestFileList[i].IsSelected = true;
                            }
                            var timgFile = timgDir + FileList[i].FileName;
                            System.Drawing.Image img = System.Drawing.Image.FromFile(oimgFile);
                            Bitmap bmp = (Bitmap)Bitmap.FromFile(oimgFile);
                            if (bmp != null && System.Drawing.Image.GetPixelFormatSize(bmp.PixelFormat) / 8 >= 3)
                                File.Copy(oimgFile, timgFile, true);
                            //else if(bmp != null && System.Drawing.Image.GetPixelFormatSize(bmp.PixelFormat) / 8 == 1)
                            //{
                            //    BitmapData data32 = bmp.LockBits(new System.Drawing.Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
                            //    Bitmap newImage = new Bitmap(bmp.Width, bmp.Height, PixelFormat.Format24bppRgb);
                            //    BitmapData data24 = newImage.LockBits(new System.Drawing.Rectangle(0, 0, newImage.Width, newImage.Height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
                            //    int offset32 = 0;
                            //    int offset24 = data24.Stride - bmp.Width * 3;
                            //    unsafe
                            //    {
                            //        byte* ptr32 = (byte*)data32.Scan0.ToPointer();
                            //        byte* ptr24 = (byte*)data24.Scan0.ToPointer();
                            //        for (int y = 0; y < bmp.Height; y++)
                            //        {
                            //            for (int x = 0; x < bmp.Width; x++)
                            //            {
                            //                ptr32++;
                            //                * ptr24++ = *ptr32++;
                            //                *ptr24++ = *ptr32++;
                            //                *ptr24++ = *ptr32++;
                            //            }
                            //            ptr32 += data32.Stride - bmp.Width;
                            //            ptr24 += offset24;
                            //        }
                            //    }
                            //    bmp.UnlockBits(data32);
                            //    newImage.UnlockBits(data24);
                            //    newImage.Save(timgFile);
                            //    //newImage.Save(timgFile, ImageFormat.Bmp);
                            //    //newImage.Save(timgFile, img.RawFormat);
                            //    bmp.Dispose();
                            //    newImage.Dispose();
                            //}
                            else if (bmp != null && System.Drawing.Image.GetPixelFormatSize(bmp.PixelFormat) / 8 == 1)
                            {
                                //Convert8BitTo24BitJpg(oimgFile, timgFile);
                                //var newImage = TransForm8to24(bmp);
                                //newImage.Save(timgFile/*.Replace(new FileInfo(timgFile).Extension.ToLower(), ".png"),ImageFormat.Png*/);
                                //newImage = null;

                                //BitmapData data8 = bmp.LockBits(new System.Drawing.Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, PixelFormat.Format8bppIndexed);
                                //Bitmap newImage = new Bitmap(bmp.Width, bmp.Height, PixelFormat.Format24bppRgb);
                                //BitmapData data24 = newImage.LockBits(new System.Drawing.Rectangle(0, 0, newImage.Width, newImage.Height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
                                //int offset8 = data8.Stride - bmp.Width;
                                //int offset24 = data24.Stride - bmp.Width * 3;
                                //unsafe
                                //{
                                //    byte* ptr8 = (byte*)data8.Scan0.ToPointer();
                                //    byte* ptr24 = (byte*)data24.Scan0.ToPointer();
                                //    for (int y = 0; y < bmp.Height; y++)
                                //    {
                                //        for (int x = 0; x < bmp.Width; x++)
                                //        {
                                //            *ptr24++ = *ptr8;
                                //            *ptr24++ = *ptr8;
                                //            *ptr24++ = *ptr8;
                                //            ptr8++;
                                //        }
                                //        ptr8 += offset8;
                                //        ptr24 += offset24;
                                //    }
                                //}
                                //bmp.UnlockBits(data8);
                                //newImage.UnlockBits(data24);
                                //newImage.Save(timgFile, ImageFormat.Jpeg);
                                ////newImage.Save(timgFile, ImageFormat.Bmp);
                                ////newImage.Save(timgFile, img.RawFormat);
                                //bmp.Dispose();
                                //newImage.Dispose();
                            }
                        }
                    }
                }
                catch { }
            }
            else
                return;
        }
        static ImageCodecInfo GetEncoderInfo(string mimeType)
        {
            ImageCodecInfo[] codecs = ImageCodecInfo.GetImageEncoders();
            for (int i = 0; i < codecs.Length; i++)
            {
                if (codecs[i].MimeType == mimeType)
                    return codecs[i];
            }
            return null;
        }
        string predictResultImagePath = "";
        private YoloTask yoloTask;
        Random rand = new Random(1024);
        private void Test_Click(object sender, RoutedEventArgs e)
        {
            string predictImagePath = mwv!.ProjectDir + @"\TestImages\";
            if (Directory.GetFiles(predictImagePath).Length == 0)
                return;
            Action<string> func = (string s) =>
            {
                richtextBox2.Dispatcher.Invoke(() =>
                {
                    richtextBox2.AppendText(s);
                    richtextBox2.ScrollToEnd();
                }, System.Windows.Threading.DispatcherPriority.Background);
            };
            predictResultImagePath = predictImagePath + @"\Result\";
            if (!Directory.Exists(predictResultImagePath))
                Directory.CreateDirectory(predictResultImagePath);
            predictResultImagePath += mwv!.MType.ToString() + @"\";
            if (!Directory.Exists(predictResultImagePath))
                Directory.CreateDirectory(predictResultImagePath);
            foreach (var file in Directory.GetFiles(predictResultImagePath))
            {
                File.Delete(file);
            }
            string ModelName = mwv!.ProjectDir + @"\Models\" + mwv!.MType.ToString() + "_" + "best.bin";
            ////Create segmenter
            if (Directory.GetFiles(predictImagePath) == null || Directory.GetFiles(predictImagePath).Length == 0||!File.Exists(ModelName))
                return;
            //window.IsExpanded = true;
            richtextBox2.Document.Blocks.Clear();
            int numClasses = Properties.Settings.Default.NumClasses;
            if (mwv!.MType == ModelType.Yolov8_Float16_n)
            {
                yoloTask = new YoloTask(TaskType.Segmentation, numClasses, yoloType: YoloType.Yolov8, deviceType:  DeviceType.CUDA, yoloSize:  YoloSize.n, dtype:  ScalarType.Float16);
            }
            else if (mwv!.MType == ModelType.Yolov8_Float32_n)
            {
                yoloTask = new YoloTask(TaskType.Segmentation, numClasses, yoloType: YoloType.Yolov8, deviceType: DeviceType.CUDA, yoloSize: YoloSize.n, dtype: ScalarType.Float32);
            }
            else if (mwv!.MType == ModelType.Yolov11_Float16_n)
            {
                yoloTask = new YoloTask(TaskType.Segmentation, numClasses, yoloType: YoloType.Yolov11, deviceType: DeviceType.CUDA, yoloSize: YoloSize.n, dtype: ScalarType.Float16);
            }
            else if (mwv!.MType == ModelType.Yolov11_Float32_n)
            {
                yoloTask = new YoloTask(TaskType.Segmentation, numClasses, yoloType: YoloType.Yolov11, deviceType: DeviceType.CUDA, yoloSize: YoloSize.n, dtype: ScalarType.Float32);
            }
            else if (mwv!.MType == ModelType.Yolov8_Float16_s)
            {
                yoloTask = new YoloTask(TaskType.Segmentation, numClasses, yoloType: YoloType.Yolov8, deviceType: DeviceType.CUDA, yoloSize: YoloSize.s, dtype: ScalarType.Float16);
            }
            else if (mwv!.MType == ModelType.Yolov8_Float32_s)
            {
                yoloTask = new YoloTask(TaskType.Segmentation, numClasses, yoloType: YoloType.Yolov8, deviceType: DeviceType.CUDA, yoloSize: YoloSize.s, dtype: ScalarType.Float32);
            }
            else if (mwv!.MType == ModelType.Yolov11_Float16_s)
            {
                yoloTask = new YoloTask(TaskType.Segmentation, numClasses, yoloType: YoloType.Yolov11, deviceType: DeviceType.CUDA, yoloSize: YoloSize.s, dtype: ScalarType.Float16);
            }
            else
            {
                yoloTask = new YoloTask(TaskType.Segmentation, numClasses, yoloType: YoloType.Yolov11, deviceType: DeviceType.CUDA, yoloSize: YoloSize.s, dtype: ScalarType.Float32);
            }
            yoloTask.LoadModel(ModelName, func, skipNcNotEqualLayers: true);
            func("\n");
            foreach (string Path in Directory.GetFiles(predictImagePath))
            {
                var fi = new FileInfo(Path);
                string PathExt = fi.Extension.ToLower();
                if (PathExt == ".png" || PathExt == ".jpg") //Json格式?
                {
                    long start = DateTime.Now.Ticks;
                    Mat predictImage = Cv2.ImRead(Path);

                    List<YoloResult> predictResult = yoloTask.ImagePredict(predictImage, mwv!.PredictThreshold, mwv!.IouThreshold);
                    if (predictResult == null)
                        return;
                    long end = DateTime.Now.Ticks;
                    func(string.Format("文件:{0}\n", fi.Name));

                    //foreach (YoloResult result in predictResult)
                    for (int i=0;i<predictResult.Count;i++)
                    {
                        YoloResult result = predictResult[i];
                        float[] cxcywhr = new float[] { result.CenterX, result.CenterY, result.Width, result.Height, result.Radian };
                        float[] points = cxcywhr2xyxyxyxy(cxcywhr);

                        Point[] pts = new Point[4]
                        {
                    new Point(points[0], points[1]),
                    new Point(points[2], points[3]),
                    new Point(points[4], points[5]),
                    new Point(points[6], points[7]),
                        };
                        Cv2.Polylines(predictImage, new Point[][] { pts }, true, Scalar.Red, 2);
                        string label = string.Format("{0}:{1:F1}%", (i+1), result.Score * 100);

                        Size textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.5, 1, out int baseline);
                        int x = 0, y = 14;
                        if (result.Y - baseline > 14)
                            y = result.Y - baseline;
                        if (result.X > 0)
                            x = result.X;

                        // Draw mask
                        if (result.Mask is not null)
                        {
                            Mat maskMat = Mat.FromArray<byte>(result.Mask);
                            maskMat = maskMat * 255;
                            //   maskMat = maskMat.Transpose();

                            // Create random color
                            int R = rand.Next(0, 255);
                            int G = rand.Next(0, 255);
                            int B = rand.Next(0, 255);
                            Scalar color = new Scalar(R, G, B, 200);
                            Mat backColor = new Mat(maskMat.Rows, maskMat.Cols, MatType.CV_8UC3, color);
                            Cv2.Add(predictImage, backColor, predictImage, maskMat);

                            OpenCvSharp.Point[][] contours;
                            Cv2.FindContours(
                                image: maskMat,
                                contours: out contours,
                                hierarchy: out HierarchyIndex[] outputArray,
                                mode: (RetrievalModes)RetrievalModes.External,
                                method: (ContourApproximationModes)ContourApproximationModes.ApproxSimple
                                );
                            List<OpenCvSharp.Point[]> query = contours.ToList<OpenCvSharp.Point[]>().OrderByDescending(t => Cv2.ContourArea(t)).Select(t => t).Take(1).ToList();
                            if (query.Count == 0)
                                return;
                            var epsilon = Cv2.ArcLength(query[0], true) * 0.003;
                            var approxContour = Cv2.ApproxPolyDP(query[0], epsilon, true);
                            var curPS = new List<OpenCvSharp.Point>();
                            for (int k = 0; k < approxContour.Length; k++)
                            {
                                var p = new OpenCvSharp.Point((double)(approxContour[k].X), (double)(approxContour[k].Y));
                                curPS.Add(p);
                                Cv2.Circle(predictImage, p, 3, Scalar.GreenYellow, 1);
                            }
                            Cv2.DrawContours(predictImage, new OpenCvSharp.Point[][] { curPS.ToArray() }, -1, Scalar.YellowGreen, 1);
                        }
                        Cv2.Rectangle(predictImage, new OpenCvSharp.Rect(new Point(x, y - textSize.Height), new Size(textSize.Width, textSize.Height + baseline)), Scalar.White, Cv2.FILLED);
                        Cv2.PutText(predictImage, label, new Point(x, y), HersheyFonts.HersheySimplex, 0.5, Scalar.Black, 1);
                        func(string.Format("  Index :{0}\n     Score:{1:F1}%\n     CenterX:{2}\n     CenterY:{3}\n     Width:{4}\n     Height:{5}\n     R:{6:F3}\n", (i+1), result.Score * 100, result.CenterX, result.CenterY, result.Width, result.Height, result.Radian));
                    }
                    string Name = new FileInfo(Path).Name;
                    predictImage.SaveImage(predictResultImagePath + Name);
                    long span = (end - start) / TimeSpan.TicksPerMillisecond;
                    func("用时："+span.ToString() + " ms\n\n");
                }
            }
            func("ImagePredict done");
            try
            {
                TFileList.Clear();
                foreach (string Path in Directory.GetFiles(predictResultImagePath))
                {
                    var PExt = new FileInfo(Path).Extension.ToLower();
                    if (PExt == ".jpg" || PExt == ".bmp" || PExt == ".png") //筛选图片格式
                    {
                        var fi = new FileInfo(Path);
                        TFileList.Add(
                            new SelectableFiles() { FileName = fi.Name, Directory = fi.Directory.FullName }
                            );
                    }
                }
                mwv!.TestResultFileList = TFileList;
                if (TFileList.Count > 0)
                {
                    TestResultFileListGrid.SelectedIndex = 0;
                }
            }
            catch { }
        }
        Bitmap sourceBitmap = null;
        private void DataGrid_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (TestResultFileListGrid.SelectedIndex > -1)
            {
                try
                {
                    var FileName = mwv!.TestResultFileList[TestResultFileListGrid.SelectedIndex].FileName;
                    var dir = mwv!.TestResultFileList[TestResultFileListGrid.SelectedIndex].Directory;
                    StreamReader streamReader = new StreamReader(dir + @"\" + FileName);
                    var OrignalBitmap = (Bitmap)Bitmap.FromStream(streamReader.BaseStream);
                    streamReader.Close();
                    int width = GWpf2.Width;
                    int height = GWpf2.Height;
                    var wratio = (float)((float)OrignalBitmap.Width / (float)width);
                    var hratio = (float)((float)OrignalBitmap.Height / (float)height);
                    sourceBitmap = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format32bppRgb);
                    sourceBitmap.SetResolution(OrignalBitmap.HorizontalResolution, OrignalBitmap.VerticalResolution);
                    Graphics graphic = Graphics.FromImage(sourceBitmap);
                    graphic.SmoothingMode = SmoothingMode.HighQuality;
                    graphic.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    graphic.DrawImage(OrignalBitmap, new System.Drawing.Rectangle(0, 0, width, height));
                    graphic.Dispose();
                    DrawToGraphics(sourceBitmap);
                }
                catch
                {
                    if (TestResultFileListGrid.SelectedIndex < TestResultFileListGrid.Items.Count - 1)
                        TestResultFileListGrid.SelectedIndex++;
                }
            }
        }
        private unsafe void DrawToGraphics(Bitmap bm)
        {
            lock (GWpf2.Lock)
            {
                GWpf2.GFX.SmoothingMode = SmoothingMode.AntiAlias;
                GWpf2.GFX.SmoothingMode = SmoothingMode.HighQuality;
                GWpf2.GFX.CompositingQuality = CompositingQuality.HighQuality;
                GWpf2.GFX.PixelOffsetMode = PixelOffsetMode.HighQuality;
                Graphics g = GWpf2.GFX;
                g.Clear(Color.White);
                if (bm != null)
                    g.DrawImage(bm, 0, 0);
                GWpf2.Paint();
            }
        }

        private void GWpf2_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            DrawToGraphics(sourceBitmap);
        }
        public void timer_Tick(object sender, EventArgs e)
        {
            //mwv?.GetLaserDataCommand.Execute(null);
            DrawToGraphics(sourceBitmap);
        }
        /// <summary>
        /// 8位转24位
        /// </summary>
        /// <param name="bmp"></param>
        /// <returns></returns>
        public static Bitmap TransForm8to24(Bitmap bmp)
        {

            System.Drawing.Rectangle rect = new System.Drawing.Rectangle(0, 0, bmp.Width, bmp.Height);
            System.Drawing.Imaging.BitmapData bitmapData = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, bmp.PixelFormat);

            //计算实际8位图容量
            int size8 = bitmapData.Stride * bmp.Height;
            byte[] grayValues = new byte[size8];

            //// 申请目标位图的变量，并将其内存区域锁定  
            Bitmap TempBmp = new Bitmap(bmp.Width, bmp.Height, PixelFormat.Format24bppRgb);
            BitmapData TempBmpData = TempBmp.LockBits(new System.Drawing.Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);


            //// 获取图像参数以及设置24位图信息 
            int stride = TempBmpData.Stride;  // 扫描线的宽度  
            int offset = stride - TempBmp.Width;  // 显示宽度与扫描线宽度的间隙  
            IntPtr iptr = TempBmpData.Scan0;  // 获取bmpData的内存起始位置  
            int scanBytes = stride * TempBmp.Height;// 用stride宽度，表示这是内存区域的大小  

            //// 下面把原始的显示大小字节数组转换为内存中实际存放的字节数组  

            byte[] pixelValues = new byte[scanBytes];  //为目标数组分配内存  
            System.Runtime.InteropServices.Marshal.Copy(bitmapData.Scan0, grayValues, 0, size8);


            for (int i = 0; i < bmp.Height; i++)
            {

                for (int j = 0; j < bitmapData.Stride; j++)
                {

                    if (j >= bmp.Width)
                        continue;
                    int indexSrc = i * bitmapData.Stride + j;
                    int realIndex = i * TempBmpData.Stride + j * 3;

                    pixelValues[realIndex] = grayValues[indexSrc];
                    pixelValues[realIndex + 1] = grayValues[indexSrc]++;
                    pixelValues[realIndex + 2] = grayValues[indexSrc]++;
                }
            }
            //// 用Marshal的Copy方法，将刚才得到的内存字节数组复制到BitmapData中  
            System.Runtime.InteropServices.Marshal.Copy(pixelValues, 0, iptr, scanBytes);
            TempBmp.UnlockBits(TempBmpData);  // 解锁内存区域  
            bmp.UnlockBits(bitmapData);
            return TempBmp;
        }

        private static float[] cxcywhr2xyxyxyxy(float[] x)
        {
            float cx = x[0];
            float cy = x[1];
            float w = x[2];
            float h = x[3];
            float r = x[4];
            float cosR = (float)Math.Cos(r);
            float sinR = (float)Math.Sin(r);
            float wHalf = w / 2;
            float hHalf = h / 2;
            return new float[]
            {
                cx - wHalf * cosR + hHalf * sinR,
                cy - wHalf * sinR - hHalf * cosR,
                cx + wHalf * cosR + hHalf * sinR,
                cy + wHalf * sinR - hHalf * cosR,
                cx + wHalf * cosR - hHalf * sinR,
                cy + wHalf * sinR + hHalf * cosR,
                cx - wHalf * cosR - hHalf * sinR,
                cy - wHalf * sinR + hHalf * cosR,
            };
        }
    }
}
