using ImageSegmentation.Domain;
using MaterialDesignColors;
using NumpyDotNet;
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
using System.Windows;
using System.Windows.Controls;
using System.Windows.Forms;
using System.Windows.Input;
using Color = System.Drawing.Color;
using ColorConverter = System.Windows.Media.ColorConverter;
using ColorW = System.Windows.Media.Color;
using Pen = System.Drawing.Pen;
using Point = System.Drawing.Point;
using Utilities;
using System.Threading.Tasks;
using YoloSharp;
using System.Reflection.Metadata;
using System.Windows.Shapes;
using static System.Net.Mime.MediaTypeNames;
using System.Windows.Documents;
using System.Threading;
using ImageMagick.Drawing;
using ImageMagick;
using System.Windows.Media.Media3D;
using Windows.Devices.Radios;
using System.Timers;

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
            Window? window = Window.GetWindow(this);
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
        static void Convert8BitTo24BitJpg(string inputPath, string outputPath)
        {
            using (Bitmap original = new Bitmap(inputPath))
            {
                // 创建一个24位的新Bitmap
                Bitmap newImage = new Bitmap(original.Width, original.Height, PixelFormat.Format24bppRgb);

                try
                {
                    // 设置DPI与原始图像一致
                    newImage.SetResolution(original.HorizontalResolution, original.VerticalResolution);

                    // 绘制图像到新的Bitmap上
                    using (Graphics g = Graphics.FromImage(newImage))
                    {
                        g.Clear(Color.White);
                        g.DrawImage(original, 0, 0, original.Width, original.Height);
                    }

                    // 设置JPG保存质量（0-100，数值越大质量越高）
                    ImageCodecInfo jpgCodec = GetEncoderInfo("image/jpeg");
                    EncoderParameters encoderParameters = new EncoderParameters(1);
                    encoderParameters.Param[0] = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, 90L);

                    // 保存为24位JPG
                    newImage.Save(outputPath, jpgCodec, encoderParameters);
                }
                finally
                {
                    // 确保资源被释放
                    newImage.Dispose();
                }
            }
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
        private void Test_Click(object sender, RoutedEventArgs e)
        {
            Action<string> func = (string s) =>
            {
                richtextBox2.Dispatcher.Invoke(() =>
                {
                    richtextBox2.AppendText(s);
                    richtextBox2.ScrollToEnd();
                }, System.Windows.Threading.DispatcherPriority.Background);
            };
            string predictImagePath = mwv!.ProjectDir + @"\TestImages\";
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
            window.IsExpanded = true;
            richtextBox2.Document.Blocks.Clear();
            Segmenter segmenter;
            if (mwv!.MType == ModelType.Yolov8_Float16_Cuda)
            {
                segmenter = new Segmenter(mwv!.SortCount, yoloType: YoloType.Yolov8, deviceType: DeviceType.CUDA, yoloSize: YoloSize.n, dtype: ScalarType.Float16);
            }
            else if (mwv!.MType == ModelType.Yolov8_Float32_Cuda)
            {
                segmenter = new Segmenter(mwv!.SortCount, yoloType: YoloType.Yolov8, deviceType: DeviceType.CUDA, yoloSize: YoloSize.n, dtype: ScalarType.Float32);
            }
            else if (mwv!.MType == ModelType.Yolov11_Float16_Cuda)
            {
                segmenter = new Segmenter(mwv!.SortCount, yoloType: YoloType.Yolov11, deviceType: DeviceType.CUDA, yoloSize: YoloSize.n, dtype: ScalarType.Float16);
            }
            else if (mwv!.MType == ModelType.Yolov11_Float32_Cuda)
            {
                segmenter = new Segmenter(mwv!.SortCount, yoloType: YoloType.Yolov11, deviceType: DeviceType.CUDA, yoloSize: YoloSize.n, dtype: ScalarType.Float32);
            }
            else if (mwv!.MType == ModelType.Yolov8_Float16_Cpu)
            {
                segmenter = new Segmenter(mwv!.SortCount, yoloType: YoloType.Yolov8, deviceType: DeviceType.CPU, yoloSize: YoloSize.n, dtype: ScalarType.Float16);
            }
            else if (mwv!.MType == ModelType.Yolov8_Float32_Cpu)
            {
                segmenter = new Segmenter(mwv!.SortCount, yoloType: YoloType.Yolov8, deviceType: DeviceType.CPU, yoloSize: YoloSize.n, dtype: ScalarType.Float32);
            }
            else if (mwv!.MType == ModelType.Yolov11_Float16_Cpu)
            {
                segmenter = new Segmenter(mwv!.SortCount, yoloType: YoloType.Yolov11, deviceType: DeviceType.CPU, yoloSize: YoloSize.n, dtype: ScalarType.Float16);
            }
            else
            {
                segmenter = new Segmenter(mwv!.SortCount, yoloType: YoloType.Yolov11, deviceType: DeviceType.CPU, yoloSize: YoloSize.n, dtype: ScalarType.Float32);
            }
            segmenter.LoadModel(ModelName);

            // ImagePredict image


            foreach (string Path in Directory.GetFiles(predictImagePath))
            {
                string PathExt = new FileInfo(Path).Extension.ToLower();
                if (PathExt == ".png" || PathExt == ".jpg") //Json格式?
                {
                    long start = DateTime.Now.Ticks;
                    MagickImage predictImage = new MagickImage(Path);
                    var (predictResult, resultImage) = segmenter.ImagePredict(predictImage, null, mwv!.PredictThreshold, mwv!.IouThreshold);

                    if (predictResult.Count > 0)
                    {
                        var drawables = new Drawables()
                            .StrokeColor(MagickColors.Red)
                            .StrokeWidth(1)
                            .FillColor(MagickColors.Transparent)
                            .Font("Consolas")
                            .FontPointSize(16)
                            .TextAlignment(ImageMagick.TextAlignment.Left);

                        foreach (var result in predictResult)
                        {
                            drawables.Rectangle(result.X, result.Y, result.X + result.W, result.Y + result.H);
                            string label = string.Format("Sort:{0}, Score:{1:F1}%", result.ClassID, result.Score * 100);
                            drawables.Text(result.X, result.Y - 12, label);
                            func(label + "\n");
                        }
                        resultImage.Draw(drawables);
                        string Name = new FileInfo(Path).Name;
                        resultImage.Write(predictResultImagePath + Name);
                    }
                    long end = DateTime.Now.Ticks;
                    long span = (end - start) / TimeSpan.TicksPerMillisecond;
                    func(span.ToString());
                    func("\n");
                }
            }
            func("\n");
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
    }
}
