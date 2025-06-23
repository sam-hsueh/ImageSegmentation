using MaterialDesignThemes.Wpf;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Windows.Data;

namespace ImageSegmentation.Domain
{
    public class MainWindowViewModel : ViewModelBase
    {
        MainWindow? WIN;
        public ObservableCollection<double[]>? Data { get; }
        Object? obj = new object();
        double[,] J;
        public Home? mainW
        {
            set;
            get;
        }
        public YModel? ymodelW
        {
            set;
            get;
        }
        public Test? testW
        {
            set;
            get;
        }

        public MainWindowViewModel()
        {
            MenuItems = new ObservableCollection<MenuItem>(new[]
           {
                new MenuItem(
                    "标注",
                    typeof(Home),
                PackIconKind.Home,
                PackIconKind.HomeOutline,
                    this
                ),
                new MenuItem(
                    "训练",
                    typeof(YModel),
                    PackIconKind.LetterYBoxOutline,
                    PackIconKind.LetterYBox,
                    this
                ),
                new MenuItem(
                    "预测",
                    typeof(Test),
                    PackIconKind.LetterTBoxOutline,
                    PackIconKind.LetterTBox,
                    this
                )
            });
            WIN = System.Windows.Application.Current.MainWindow as MainWindow;
            HomeCommand = new AnotherCommandImplementation(_ => { SelectedItem = MenuItems[0]; });
            YModelCommand = new AnotherCommandImplementation(_ => { SelectedItem = MenuItems[1]; });
            TestCommand = new AnotherCommandImplementation(_ => { SelectedItem = MenuItems[2]; });
            TrainCommand = new AnotherCommandImplementation(Train);
            SelectedItem = MenuItems[SelectedIndex];
            _menuItemsView = CollectionViewSource.GetDefaultView(MenuItems);
            FeatureList = new ObservableCollection<SelectableFeature>();
            var t2 = new System.Windows.Forms.Timer();
            t2.Interval = 500;
            t2.Tick += new System.EventHandler(CheckModelExits);//Get Robot Data
            t2.Start();

        }

        private void CheckModelExits(object? sender, EventArgs e)
        {
            string M = ProjectDir + @"\Models\" + Enum.GetValues(typeof(ModelType)).GetValue(0).ToString() + "_best.bin";
            Yolov8_Float16_Cuda= File.Exists(M);
            M = ProjectDir + @"\Models\" + Enum.GetValues(typeof(ModelType)).GetValue(1).ToString() + "_best.bin";
            Yolov8_Float32_Cuda = File.Exists(M);
            M = ProjectDir + @"\Models\" + Enum.GetValues(typeof(ModelType)).GetValue(2).ToString() + "_best.bin";
            Yolov11_Float16_Cuda = File.Exists(M);
            M = ProjectDir + @"\Models\" + Enum.GetValues(typeof(ModelType)).GetValue(3).ToString() + "_best.bin";
            Yolov11_Float32_Cuda = File.Exists(M);
            M = ProjectDir + @"\Models\" + Enum.GetValues(typeof(ModelType)).GetValue(4).ToString() + "_best.bin";
            Yolov8_Float16_Cpu = File.Exists(M);
            M = ProjectDir + @"\Models\" + Enum.GetValues(typeof(ModelType)).GetValue(5).ToString() + "_best.bin";
            Yolov8_Float32_Cpu = File.Exists(M);
            M = ProjectDir + @"\Models\" + Enum.GetValues(typeof(ModelType)).GetValue(6).ToString() + "_best.bin";
            Yolov11_Float16_Cpu = File.Exists(M);
            M = ProjectDir + @"\Models\" + Enum.GetValues(typeof(ModelType)).GetValue(7).ToString() + "_best.bin";
            Yolov11_Float32_Cpu = File.Exists(M);
        }

        private void Train(object? obj)
        {
            string trainDataPath = ProjectDir + @"\DataSets";
            string valDataPath = string.Empty;
            string outputPath = ProjectDir + @"\Models";
            string preTrainedModelPath = AppDomain.CurrentDomain.BaseDirectory + @"\PreTrainedModels\";
            if (YType == YoloType.Yolov8)
            {
                if (SType == ScalarType.Float16)
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
                if (SType == ScalarType.Float16)
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

            ////YoloType yoloType = YoloType.Yolov11;
            //DeviceType deviceType = DeviceType.CUDA;
            ////ScalarType dtype = ScalarType.Float32;
            //YoloSize yoloSize = YoloSize.n;

            ////Task.Factory.StartNew(()=>
            ////{
            //    //Create segmenter
            //    Segmenter segmenter = new Segmenter(SortCount, yoloType: YType, deviceType: DType, yoloSize: yoloSize, dtype: SType);
            //    segmenter.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);

            //    // Train model
            //    segmenter.Train(trainDataPath, valDataPath, outputPath: outputPath, batchSize: BatchSize, epochs: Epochs, useMosaic: false);
            ////});
        }

        public ObservableCollection<MenuItem> MenuItems { get; }
        public ObservableCollection<MenuItem> MainMenuItems { get; }

        public MenuItem? SelectedItem
        {
            get => _selectedItem;
            set => SetProperty(ref _selectedItem, value);
        }

        public int SelectedIndex
        {
            get => _selectedIndex;
            set => SetProperty(ref _selectedIndex, value);
        }

        public bool ControlsEnabled
        {
            get => _controlsEnabled;
            set => SetProperty(ref _controlsEnabled, value);
        }
        private string _text;
        public string Text
        {
            get => _text;
            set => SetProperty(ref _text, value);
        }

        public AnotherCommandImplementation HomeCommand { get; }
        public AnotherCommandImplementation YModelCommand { get; }
        //public AnotherCommandImplementation DoCamCommand { get; }
        //public AnotherCommandImplementation WeldingCommand { get; }
        public AnotherCommandImplementation TestCommand { get; }
        public AnotherCommandImplementation SettingsCommand { get; }
        public AnotherCommandImplementation SaveSettingCommand { get; }
        public AnotherCommandImplementation TrainCommand { get; }
        public AnotherCommandImplementation CloseRoboCommCommand { get; }
        public AnotherCommandImplementation? ComputeCommand { get; }
        public AnotherCommandImplementation? SingularRoboCommand { get; }
        public AnotherCommandImplementation? ResetRoboCommand { get; }
        public AnotherCommandImplementation? OriginalRoboCommand { get; }
        //public AnotherCommandImplementation? SimuCommand { get; }
        public AnotherCommandImplementation? StartCommand { get; }
        public AnotherCommandImplementation? ReStartCommand { get; }
        public AnotherCommandImplementation? NextCommand { get; }
        public AnotherCommandImplementation? QuitCommand { get; }
        public AnotherCommandImplementation? ResetCommand { get; }
        public AnotherCommandImplementation? SetPRCommand { get; }
        public AnotherCommandImplementation? SendPCommand { get; }
        public AnotherCommandImplementation? HelpCommand { get; }
        public AnotherCommandImplementation? RightRoboCommand { get; }
        public AnotherCommandImplementation? ForwardCommand { get; }
        public AnotherCommandImplementation? GetLaserDataCommand { get; }
        public AnotherCommandImplementation? OpenLaserLightCommand { get; }

        public AnotherCommandImplementation? CloseLaserLightCommand { get; }
        public AnotherCommandImplementation? ConnectLaserCommCommand { get; }
        public AnotherCommandImplementation? CloseLaserCommCommand { get; }
        private readonly ICollectionView? _menuItemsView;
        private MenuItem? _selectedItem;
        private int _selectedIndex;
        private bool _controlsEnabled = true;

        ObservableCollection<SelectableFiles> _FileList;
        public ObservableCollection<SelectableFiles> FileList
        {
            set => SetProperty(ref _FileList, value);
            get => _FileList;
        }
        ObservableCollection<SelectableFeature> _FeatureList;
        public ObservableCollection<SelectableFeature> FeatureList
        {
            set => SetProperty(ref _FeatureList, value);
            get => _FeatureList;
        }
        public ObservableCollection<int> CatFeatures
        {
            get
            {
                var arr = new ObservableCollection<int>();
                for(int i=0;i<100;i++)
                    arr.Add(i);
                return arr;
            }
        }

        string? _ProjectDir = string.Empty;
        public string? ProjectDir
        {
            set => SetProperty(ref _ProjectDir, value);
            get => _ProjectDir;
        }

        string? _OriginalImageDir = string.Empty;
        public string? OriginalImageDir
        {
            set => SetProperty(ref _OriginalImageDir, value);
            get => _OriginalImageDir;
        }
        ObservableCollection<SelectableFiles> _TestFileList;
        public ObservableCollection<SelectableFiles> TestFileList
        {
            set => SetProperty(ref _TestFileList, value);
            get => _TestFileList;
        }
        ObservableCollection<SelectableFiles> _TestResultFileList;
        public ObservableCollection<SelectableFiles> TestResultFileList
        {
            set => SetProperty(ref _TestResultFileList, value);
            get => _TestResultFileList;
        }

        int _BatchSize = 16;
        public int BatchSize
        {
            set => SetProperty(ref _BatchSize, value);
            get => _BatchSize;
        }

        int _SortCount = 80;
        public int SortCount
        {
            set => SetProperty(ref _SortCount, value);
            get => _SortCount;
        }
        int _Epochs = 100;
        public int Epochs
        {
            set => SetProperty(ref _Epochs, value);
            get => _Epochs;
        }

        float _PredictThreshold = 0.25f;
        public float PredictThreshold
        {
            set => SetProperty(ref _PredictThreshold, value);
            get => _PredictThreshold;
        }
        float _IouThreshold = 0.7f;
        public float IouThreshold
        {
            set => SetProperty(ref _IouThreshold, value);
            get => _IouThreshold;
        }
        YoloType _YType = YoloType.Yolov11;
        public YoloType YType
        {
            set => SetProperty(ref _YType, value);
            get => _YType;
        }

        DeviceType _DType = DeviceType.CPU;
        public DeviceType DType
        {
            set => SetProperty(ref _DType, value);
            get => _DType;
        }

        ScalarType _SType = ScalarType.Float32;
        public ScalarType SType
        {
            set => SetProperty(ref _SType, value);
            get => _SType;
        }
        ModelType _MType = ModelType.Yolov8_Float16_Cuda;
        public ModelType MType
        {
            set => SetProperty(ref _MType, value);
            get => _MType;
        }
        bool _Yolov8_Float16_Cuda = false;
        public bool Yolov8_Float16_Cuda
        {
            set => SetProperty(ref _Yolov8_Float16_Cuda, value);
            get => _Yolov8_Float16_Cuda;
        }
        bool _Yolov8_Float32_Cuda = false;
        public bool Yolov8_Float32_Cuda
        {
            set => SetProperty(ref _Yolov8_Float32_Cuda, value);
            get => _Yolov8_Float32_Cuda;
        }
        bool _Yolov11_Float16_Cuda = false;
        public bool Yolov11_Float16_Cuda
        {
            set => SetProperty(ref _Yolov11_Float16_Cuda, value);
            get => _Yolov11_Float16_Cuda;
        }
        bool _Yolov11_Float32_Cuda = false;
        public bool Yolov11_Float32_Cuda
        {
            set => SetProperty(ref _Yolov11_Float32_Cuda, value);
            get => _Yolov11_Float32_Cuda;
        }
        bool _Yolov8_Float16_Cpu = false;
        public bool Yolov8_Float16_Cpu
        {
            set => SetProperty(ref _Yolov8_Float16_Cpu, value);
            get => _Yolov8_Float16_Cpu;
        }
        bool _Yolov8_Float32_Cpu = false;
        public bool Yolov8_Float32_Cpu
        {
            set => SetProperty(ref _Yolov8_Float32_Cpu, value);
            get => _Yolov8_Float32_Cpu;
        }
        bool _Yolov11_Float16_Cpu = false;
        public bool Yolov11_Float16_Cpu
        {
            set => SetProperty(ref _Yolov11_Float16_Cpu, value);
            get => _Yolov11_Float16_Cpu;
        }
        bool _Yolov11_Float32_Cpu = false;
        public bool Yolov11_Float32_Cpu
        {
            set => SetProperty(ref _Yolov11_Float32_Cpu, value);
            get => _Yolov11_Float32_Cpu;
        }
    }
    public enum ModelType
    {
        Yolov8_Float16_Cuda = 0,
        Yolov8_Float32_Cuda = 1,
        Yolov11_Float16_Cuda = 2,
        Yolov11_Float32_Cuda = 3,
        Yolov8_Float16_Cpu = 4,
        Yolov8_Float32_Cpu = 5,
        Yolov11_Float16_Cpu = 6,
        Yolov11_Float32_Cpu = 7,

    }
}
