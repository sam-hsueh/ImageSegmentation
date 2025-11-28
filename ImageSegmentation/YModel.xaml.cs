using System;
using System.Threading.Tasks;
using System.Windows;
using ImageSegmentation.Domain;
using YoloSharp.Models;
using YoloSharp.Types;

namespace ImageSegmentation
{
    /// <summary>
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
            string rootPath = mwv!.ProjectDir + @"\DataSets";
            string valDataPath = string.Empty;
            string outputPath = mwv!.ProjectDir + @"\Models";
            string preTrainedModelPath = AppDomain.CurrentDomain.BaseDirectory + @"\PreTrainedModels\";
            if (mwv!.YType == YoloType.Yolov8)
            {
                if (mwv!.YSize == YoloSize.n)
                    preTrainedModelPath += @"yolov8n-seg.bin";

                else
                    preTrainedModelPath += @"yolov8s-seg.bin";
            }
            else
            {
                if (mwv!.YSize == YoloSize.n)
                    preTrainedModelPath += @"yolov11n-seg.bin";

                else
                    preTrainedModelPath += @"yolov11s-seg.bin";
            }
            int numClasses = Properties.Settings.Default.NumClasses;
            //Create segmenter
            YoloTask segmenter = new YoloTask(TaskType.Segmentation, numClasses, yoloType: mwv!.YType, deviceType: DeviceType.CUDA, yoloSize: mwv!.YSize, dtype: mwv!.SType);
            Action<string> func = (string s) =>
            {
                richtextBox1.Dispatcher.Invoke(() =>
                {
                    richtextBox1.AppendText(s);
                    richtextBox1.ScrollToEnd();
                }, System.Windows.Threading.DispatcherPriority.Background);
            };
                segmenter.LoadModel(preTrainedModelPath,func, skipNcNotEqualLayers: true);

            Task.Run(() =>
            {
                // Train model
                segmenter.Train(rootPath: rootPath,func: func, valDataPath:valDataPath, outputPath: outputPath, batchSize: mwv!.BatchSize, epochs: mwv!.Epochs,imageProcessType: ImageProcessType.Mosiac);
            });
        }
    }
}
