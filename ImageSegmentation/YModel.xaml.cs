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

namespace ImageSegmentation
{
    /// <summary>
    /// Interaction logic for MainWindoRWidth.xaml
    /// </summary>
    public partial class YModel : System.Windows.Controls.UserControl
    {
        MainWindowViewModel? mwv;
        public YModel()
        {
            InitializeComponent();
            Loaded += YModel_Loaded;
        }

        private void YModel_Loaded(object sender, RoutedEventArgs e)
        {
            Window? window = Window.GetWindow(this);
            mwv = this.DataContext as MainWindowViewModel;
            mwv!.ymodelW = this;
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            richtextBox1.Document.Blocks.Clear();
            if (mwv == null || mwv!.ProjectDir == string.Empty)
                return;
            string trainDataPath = mwv!.ProjectDir + @"\DataSets";
            string valDataPath = string.Empty;
            string outputPath = mwv!.ProjectDir + @"\Models";
            string preTrainedModelPath = AppDomain.CurrentDomain.BaseDirectory + @"\PreTrainedModels\";
            if (mwv!.YType == YoloType.Yolov8)
            {
                if (mwv!.SType == ScalarType.Float16)
                {
                    preTrainedModelPath += @"F16\yolov8n-seg.bin";
                }
                else
                {
                    preTrainedModelPath += @"F32\yolov8n-seg.bin";
                }
            }
            else
            {
                if (mwv!.SType == ScalarType.Float16)
                {
                    preTrainedModelPath += @"F16\yolov11n-seg.bin";
                }
                else
                {
                    preTrainedModelPath += @"F32\yolov11n-seg.bin";
                }
            }
            //int batchSize = 16;
            //int sortCount = 80;
            //int epochs = 100;
            //float predictThreshold = 0.25f;
            //float iouThreshold = 0.7f;
            //YoloType yoloType = YoloType.Yolov11;
            //ScalarType dtype = ScalarType.Float32;

            //DeviceType deviceType = DeviceType.CPU;
            YoloSize yoloSize = YoloSize.n;

            //Create segmenter
            Segmenter segmenter = new Segmenter(mwv!.SortCount, yoloType: mwv!.YType, deviceType: mwv!.DType, yoloSize: yoloSize, dtype: mwv!.SType);
            segmenter.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);

            Action<string> func = (string s) =>
            {
                richtextBox1.Dispatcher.Invoke(() =>
                {
                    richtextBox1.AppendText(s);
                    richtextBox1.ScrollToEnd();
                }, System.Windows.Threading.DispatcherPriority.Background);
            };
            Task.Run(() =>
            {
                // Train model
                segmenter.Train(trainDataPath, func, valDataPath, outputPath: outputPath, batchSize: mwv!.BatchSize, epochs: mwv!.Epochs, useMosaic: false);
            });
        }
    }
}
