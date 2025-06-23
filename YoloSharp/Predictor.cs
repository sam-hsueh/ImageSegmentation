using System.Text.RegularExpressions;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static YoloSharp.Yolo;

namespace YoloSharp
{
	public class Predictor
	{
		public class PredictResult
		{
			public int ClassID;
			public float Score;
			public int X;
			public int Y;
			public int W;
			public int H;
		}

		private Module<Tensor, Tensor[]> yolo;
		private Module<Tensor[], Tensor, (Tensor, Tensor)> loss;
		private Module<Tensor, float, float, Tensor> predict;
		private torch.Device device;
		private torch.ScalarType dtype;
		private int socrCount;
		private YoloType yoloType;

		public Predictor(int socrCount = 80, YoloType yoloType = YoloType.Yolov8, YoloSize yoloSize = YoloSize.n, DeviceType deviceType = DeviceType.CUDA, ScalarType dtype = ScalarType.Float32)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

			this.device = new torch.Device((TorchSharp.DeviceType)deviceType);
			this.dtype = (torch.ScalarType)dtype;
			this.socrCount = socrCount;
			this.yoloType = yoloType;

			yolo = yoloType switch
			{
				YoloType.Yolov5 => new Yolov5(socrCount, yoloSize, device, this.dtype),
				YoloType.Yolov5u => new Yolov5u(socrCount, yoloSize, device, this.dtype),
				YoloType.Yolov8 => new Yolov8(socrCount, yoloSize, device, this.dtype),
				YoloType.Yolov11 => new Yolov11(socrCount, yoloSize, device, this.dtype),
				YoloType.Yolov12 => new Yolov12(socrCount, yoloSize, device, this.dtype),
				_ => throw new NotImplementedException(),
			};

			loss = yoloType switch
			{
				YoloType.Yolov5 => new Loss.Yolov5DetectionLoss(this.socrCount),
				YoloType.Yolov5u => new Loss.YoloDetectionLoss(this.socrCount),
				YoloType.Yolov8 => new Loss.YoloDetectionLoss(this.socrCount),
				YoloType.Yolov11 => new Loss.YoloDetectionLoss(this.socrCount),
				YoloType.Yolov12 => new Loss.YoloDetectionLoss(this.socrCount),
				_ => throw new NotImplementedException(),
			};
			predict = yoloType switch
			{
				YoloType.Yolov5 => new Predict.Yolov5Predict(),
				YoloType.Yolov5u => new Predict.YoloPredict(),
				YoloType.Yolov8 => new Predict.YoloPredict(),
				YoloType.Yolov11 => new Predict.YoloPredict(),
				YoloType.Yolov12 => new Predict.YoloPredict(),
				_ => throw new NotImplementedException(),
			};

			//Tools.TransModelFromSafetensors(yolo, @".\yolov11n.safetensors", @".\yolov11n.bin");
		}

		public void Train(string trainDataPath,Action<string>? func=null, string valDataPath = "", string outputPath = "output", int imageSize = 640, int epochs = 100, float lr = 0.0001f, int batchSize = 8, int numWorkers = 0, bool useMosaic = true)
		{
			Console.WriteLine("Model will be write to: " + outputPath + @"\n");
			Console.WriteLine("Load model...\n");

			YoloDataset trainDataSet = new YoloDataset(trainDataPath, imageSize, deviceType: this.device.type, useMosaic: useMosaic);
			if (trainDataSet.Count == 0)
			{
				throw new FileNotFoundException("No data found in the path: " + trainDataPath);
			}
			DataLoader trainDataLoader = new DataLoader(trainDataSet, batchSize, num_worker: numWorkers, shuffle: true, device: device);
			valDataPath = string.IsNullOrEmpty(valDataPath) ? trainDataPath : valDataPath;
			Optimizer optimizer = new SGD(yolo.parameters(), lr: lr);


			float tempLoss = float.MaxValue;
			func("Start Training...\n");
			yolo.train(true);
			for (int epoch = 0; epoch < epochs; epoch++)
			{
				int step = 0;
				foreach (var data in trainDataLoader)
				{
					step++;
					long[] indexs = data["index"].data<long>().ToArray();
					Tensor[] images = new Tensor[indexs.Length];
					Tensor[] labels = new Tensor[indexs.Length];
					for (int i = 0; i < indexs.Length; i++)
					{
						var (img, lb) = trainDataSet.GetDataTensor(indexs[i]);
						images[i] = img.to(dtype, device);
						labels[i] = full(new long[] { lb.shape[0], lb.shape[1] + 1 }, i, dtype: dtype, device: lb.device);
						labels[i].slice(1, 1, lb.shape[1] + 1, 1).copy_(lb);
					}
					Tensor imageTensor = concat(images);
					Tensor labelTensor = concat(labels);
					if (labelTensor.shape[0] == 0)
					{
						continue;
					}
					Tensor[] list = yolo.forward(imageTensor);
					var (ls, ls_item) = loss.forward(list, labelTensor);
					optimizer.zero_grad();
					ls.backward();
					optimizer.step();
					func($"Process: Epoch {epoch}, Step/Total Step  {step}/{trainDataLoader.Count}\n");
				}
				Console.Write("Do val now... \n");
				float valLoss = Val(valDataPath, imageSize);
				func($"Epoch {epoch}, Val Loss: {valLoss}\n");
				if (!Directory.Exists(outputPath))
				{
					Directory.CreateDirectory(outputPath);
				}
                yolo.save(Path.Combine(outputPath, yoloType.ToString() + "_" + dtype.ToString() + "_" + device.ToString() + "_last.bin"));
                if (tempLoss > valLoss)
                {
                    yolo.save(Path.Combine(outputPath, yoloType.ToString() + "_" + dtype.ToString() + "_" + device.ToString() + "_best.bin"));
                    tempLoss = valLoss;
                }
            }
            func("Train Done.\n");
		}

		private float Val(string valDataPath, int imageSize = 640)
		{
			YoloDataset yoloDataset = new YoloDataset(valDataPath, imageSize, deviceType: this.device.type, useMosaic: false);
			DataLoader loader = new DataLoader(yoloDataset, 4, num_worker: 0, shuffle: true, device: device);

			float lossValue = float.MaxValue;
			foreach (var data in loader)
			{
				long[] indexs = data["index"].data<long>().ToArray();
				Tensor[] images = new Tensor[indexs.Length];
				Tensor[] labels = new Tensor[indexs.Length];
				for (int i = 0; i < indexs.Length; i++)
				{
					var (img, lb) = yoloDataset.GetDataTensor(indexs[i]);
					images[i] = img.to(this.dtype, device);
					labels[i] = full(new long[] { lb.shape[0], lb.shape[1] + 1 }, i, dtype: dtype, device: lb.device);
					labels[i].slice(1, 1, lb.shape[1] + 1, 1).copy_(lb);
				}
				Tensor imageTensor = concat(images);
				Tensor labelTensor = concat(labels);

				if (labelTensor.shape[0] == 0)
				{
					continue;
				}

				Tensor[] list = yolo.forward(imageTensor);
				var (ls, ls_item) = loss.forward(list.ToArray(), labelTensor);
				if (lossValue == float.MaxValue)
				{
					lossValue = ls.ToSingle();
				}
				else
				{
					lossValue = lossValue + ls.ToSingle();
				}
			}
			lossValue = lossValue / yoloDataset.Count;
			return lossValue;
		}

		public List<PredictResult> ImagePredict(ImageMagick.MagickImage image, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
		{
			Tensor orgImage = Lib.GetTensorFromImage(image).to(dtype, device);
			orgImage = torch.stack(new Tensor[] { orgImage[2], orgImage[1], orgImage[0] }, dim: 0).unsqueeze(0) / 255.0f;
			int w = (int)orgImage.shape[3];
			int h = (int)orgImage.shape[2];
			int padHeight = 32 - (int)(orgImage.shape[2] % 32);
			int padWidth = 32 - (int)(orgImage.shape[3] % 32);

			padHeight = padHeight == 32 ? 0 : padHeight;
			padWidth = padWidth == 32 ? 0 : padWidth;

			Tensor input = torch.nn.functional.pad(orgImage, new long[] { 0, padWidth, 0, padHeight }, PaddingModes.Zeros);

			yolo.eval();
			Tensor[] tensors = yolo.forward(input);

			Tensor results = predict.forward(tensors[0], PredictThreshold, IouThreshold);
			List<PredictResult> predResults = new List<PredictResult>();
			for (int i = 0; i < results.shape[0]; i++)
			{
				int x = results[i, 0].ToInt32();
				int y = results[i, 1].ToInt32();
				int rw = (results[i, 2].ToInt32() - x);
				int rh = (results[i, 3].ToInt32() - y);

				float score = results[i, 4].ToSingle();
				int sort = results[i, 5].ToInt32();

				predResults.Add(new PredictResult()
				{
					ClassID = sort,
					Score = score,
					X = x,
					Y = y,
					W = rw,
					H = rh
				});
			}
			return predResults;
		}

		/// <summary>
		/// Load model from path.
		/// </summary>
		/// <param name="path">The Model path</param>
		/// <param name="skipNcNotEqualLayers">If nc not equals the label count in model, please set it true otherwise set it false.</param>
		public void LoadModel(string path,Action<string>? func=null, bool skipNcNotEqualLayers = false)
		{
			Dictionary<string, Tensor> state_dict = Lib.LoadModel(path, skipNcNotEqualLayers);

			if (state_dict.Count != yolo.state_dict().Count)
			{
				Console.WriteLine("Mismatched tensor count while loading. Make sure that the model you are loading into is exactly the same as the origin.\n");
				Console.WriteLine("Model will run with random weight.\n");
			}
			else
			{
				torch.ScalarType modelType = state_dict.Values.First().dtype;
				List<string> skipList = new List<string>();
				long nc = 0;

				if (skipNcNotEqualLayers)
				{
					string? layerPattern = yoloType switch
					{
						YoloType.Yolov5 => @"model\.24\.m",
						YoloType.Yolov5u => @"model\.24\.cv3",
						YoloType.Yolov8 => @"model\.22\.cv3",
						YoloType.Yolov11 => @"model\.23\.cv3",
						YoloType.Yolov12 => @"model\.21\.cv3",
						_ => null,
					};

					if (layerPattern != null)
					{
						skipList = state_dict.Keys.Where(x => Regex.IsMatch(x, layerPattern)).ToList();
						nc = yoloType switch
						{
							YoloType.Yolov5 => state_dict[skipList[0]].shape[0] / 3 - 5,
							_ => state_dict[skipList.LastOrDefault(a => a.EndsWith(".bias"))!].shape[0]
						};
					}

					if (nc == socrCount)
					{
						skipList.Clear();
					}
				}

				var (miss, err) = yolo.load_state_dict(state_dict, skip: skipList);
				if (skipList.Count > 0)
				{
					Console.WriteLine("Waring! You are skipping nc reference layers.\n");
					Console.WriteLine("This will get wrong result in Predict, sort count loaded in weight is " + nc + "\n");
				}
			}
		}

	}
}
