using Data;
using ImageFeatureDetection.Domain;
using System;
using System.Threading.Tasks;
using System.Windows;
using YoloSharp.Models;
using YoloSharp.Types;

namespace ImageFeatureDetection
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
            string preTrainedModelPath = AppDomain.CurrentDomain.BaseDirectory + @"PreTrainedModels\";
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
            var config = new Config
            {
                DeviceType = DeviceType.CUDA,
                ScalarType = mwv!.SType,
                RootPath = rootPath,
                TrainDataPath = string.Empty,
                ValDataPath = valDataPath,
                OutputPath = outputPath,
                YoloType = mwv!.YType,
                YoloSize = mwv!.YSize,
                TaskType = TaskType.Segmentation,
                ImageProcessType = ImageProcessType.Mosiac,
                //ImageSize = 640,
                BatchSize = mwv!.BatchSize,
                NumberClass = 80,
                PredictThreshold = 0.3f,
                IouThreshold = 0.7f,
                Workers = 4,
                Epochs = mwv!.Epochs,
            };
            YoloTask segmenter = new YoloTask(config);
            Action<string> func = (string s) =>
            {
                richtextBox1.Dispatcher.Invoke(() =>
                {
                    richtextBox1.AppendText(s);
                    richtextBox1.ScrollToEnd();
                }, System.Windows.Threading.DispatcherPriority.Background);
            };
            segmenter.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true, func);
           //    segmenter.LoadModel("d:\\ppp.bin",func, skipNcNotEqualLayers: true);

            Task.Run(() =>
            {
                // Train model
                segmenter.Train(func);
            });
        }
    }
}
