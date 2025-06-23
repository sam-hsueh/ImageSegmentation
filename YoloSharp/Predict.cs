using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace YoloSharp
{
	internal class Predict
	{
		public class Yolov5Predict : Module<Tensor, float, float, Tensor>
		{
			public Yolov5Predict() : base(nameof(Yolov5Predict))
			{

			}

			public override Tensor forward(Tensor tensor, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
			{
				var re = NonMaxSuppression(tensor, PredictThreshold, IouThreshold);

				if (!Equals(re[0], null))
				{
					return re[0];
				}
				else
				{
					return torch.tensor(new float[0, 6]);
				}
			}

			private List<Tensor> NonMaxSuppression(Tensor prediction, float confThreshold = 0.25f, float iouThreshold = 0.45f, bool agnostic = false, int max_det = 300, int nm = 0)
			{
				using (NewDisposeScope())
				{
					// Checks
					if (confThreshold < 0 || confThreshold > 1)
					{
						throw new ArgumentException($"Invalid Confidence threshold {confThreshold}, valid values are between 0.0 and 1.0");
					}
					if (iouThreshold < 0 || iouThreshold > 1)
					{
						throw new ArgumentException($"Invalid IoU {iouThreshold}, valid values are between 0.0 and 1.0");
					}

					var device = prediction.device;
					var scalType = prediction.dtype;

					var bs = prediction.shape[0]; // batch size
					var nc = prediction.shape[2] - nm - 5; // number of classes
					var xc = prediction[TensorIndex.Ellipsis, 4] > confThreshold; // candidates

					// Settings
					var max_wh = 7680; // maximum box width and height
					var max_nms = 30000; // maximum number of boxes into torchvision.ops.nms()
					var time_limit = 0.5f + 0.05f * bs; // seconds to quit after

					var t = DateTime.Now;
					var mi = 5 + nc; // mask start index
					var output = new List<Tensor>(new Tensor[bs]);
					for (int xi = 0; xi < bs; xi++)
					{
						var x = prediction[xi];
						x = x[xc[xi]]; // confidence

						// Compute conf
						x[TensorIndex.Ellipsis, TensorIndex.Slice(5, mi)] *= x[TensorIndex.Ellipsis, 4].unsqueeze(-1); // conf = obj_conf * cls_conf

						// Box/Mask
						var box = torchvision.ops.box_convert(x[TensorIndex.Ellipsis, TensorIndex.Slice(0, 4)], torchvision.ops.BoxFormats.cxcywh, torchvision.ops.BoxFormats.xyxy); // center_x, center_y, width, height) to (x1, y1, x2, y2)

						// Detections matrix nx6 (xyxy, conf, cls)

						var conf = x[TensorIndex.Colon, TensorIndex.Slice(5, mi)].max(1, true);
						var j = conf.indexes;
						x = torch.cat(new Tensor[] { box, conf.values, j.to_type(scalType) }, 1)[conf.values.view(-1) > confThreshold];

						var n = x.shape[0]; // number of boxes
						if (n == 0)
						{
							continue;
						}

						x = x[x[TensorIndex.Ellipsis, 4].argsort(descending: true)][TensorIndex.Slice(0, max_nms)]; // sort by confidence and remove excess boxes

						// Batched NMS
						var c = x[TensorIndex.Ellipsis, 5].unsqueeze(-1) * (agnostic ? 0 : max_wh); // classes
						var boxes = x[TensorIndex.Ellipsis, TensorIndex.Slice(0, 4)] + c;
						var scores = x[TensorIndex.Ellipsis, 4];
						var i = torchvision.ops.nms(boxes, scores, iouThreshold); // NMS
						i = i[TensorIndex.Slice(0, max_det)]; // limit detections

						output[xi] = x[i];

						if ((DateTime.Now - t).TotalSeconds > time_limit)
						{
							Console.WriteLine($"WARNING ⚠️ NMS time limit {time_limit:F3}s exceeded");
							break; // time limit exceeded
						}
					}

					for (int i = 0; i < output.Count; i++)
					{
						output[i] = output[i].MoveToOuterDisposeScope();
					}
					return output;
				}
			}
		}

		public class YoloPredict : Module<Tensor, float, float, Tensor>
		{
			public YoloPredict() : base(nameof(YoloPredict))
			{

			}

			public override Tensor forward(Tensor tensor, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
			{
				var re = NonMaxSuppression(tensor, PredictThreshold, IouThreshold);

				if (!Equals(re[0], null))
				{
					return re[0];
				}
				else
				{
					return torch.tensor(new float[0, 6]);
				}
			}

			private List<Tensor> NonMaxSuppression(Tensor prediction, float confThreshold = 0.25f, float iouThreshold = 0.45f, bool agnostic = false, int max_det = 300, int nm = 0)
			{
				using (NewDisposeScope())
				{
					// Checks
					if (confThreshold < 0 || confThreshold > 1)
					{
						throw new ArgumentException($"Invalid Confidence threshold {confThreshold}, valid values are between 0.0 and 1.0");
					}
					if (iouThreshold < 0 || iouThreshold > 1)
					{
						throw new ArgumentException($"Invalid IoU {iouThreshold}, valid values are between 0.0 and 1.0");
					}

					long bs = prediction.shape[0]; // batch size
					long nc = prediction.shape[1] - nm - 4; // number of classes
					long mi = 4 + nc; // mask start index
					Tensor xc = prediction[.., 4..(int)mi].amax(1) > confThreshold; // candidates

					List<Tensor> xindsList = new List<Tensor>();
					for (int idx = 0; idx < xc.shape[0]; idx++)
					{
						Tensor mask = xc[idx];
						xindsList.Add(torch.arange(mask.NumberOfElements, device: prediction.device));
					}
					Tensor xinds = torch.stack(xindsList, 0).unsqueeze(-1); // [batch, N, 1]

					prediction = prediction.transpose(-1, -2);
					// Settings
					int max_wh = 7680; // maximum box width and height
					int max_nms = 30000; // maximum number of boxes into torchvision.ops.nms()
					float time_limit = 0.5f + 0.05f * bs; // seconds to quit after

					prediction[TensorIndex.Ellipsis, ..4] =torchvision.ops.box_convert (prediction[TensorIndex.Ellipsis, ..4], torchvision.ops.BoxFormats.cxcywh, torchvision.ops.BoxFormats.xyxy);
					DateTime t = DateTime.Now;

					List<Tensor> output = new List<Tensor>(new Tensor[bs]);
					List<Tensor> keepi = new List<Tensor>(new Tensor[bs]);
					for (int xi = 0; xi < bs; xi++)
					{
						Tensor x = prediction[xi];
						Tensor xk = xinds[xi];
						Tensor filt = xc[xi];
						x = x[filt]; // confidence
						xk = xk[filt];

						Tensor[] box_cls_mask = x.split(new long[] { 4, nc, nm }, 1);
						Tensor box = box_cls_mask[0];
						Tensor cls = box_cls_mask[1];
						Tensor mask = box_cls_mask[2];

						(Tensor conf, Tensor j) = cls.max(1, keepdim: true);
						filt = conf.view(-1) > confThreshold;

						x = torch.cat(new Tensor[] { box, conf, j.@float(), mask }, 1)[filt];
						xk = xk[filt];

						// Check shape
						long n = x.shape[0];  // number of boxes
						if (n is 0)
						{
							continue;
						}

						if (n > max_nms)//  # excess boxes
						{
							filt = x[.., 4].argsort(descending: true)[..max_nms];  // sort by confidence and remove excess boxes
							(x, xk) = (x[filt], xk[filt]);
						}

						// Batched NMS
						Tensor c = x[.., 5..6] * max_wh;  // classes
						Tensor scores = x[.., 4];  // scores
						Tensor boxes = x[.., ..4] + c;  // boxes (offset by class)
						Tensor i = torchvision.ops.nms(boxes, scores, iouThreshold);  // NMS
						i = i[..max_det]; // limit detections
						(output[xi], keepi[xi]) = (x[i], xk[i].reshape(-1));
						if ((DateTime.Now - t).TotalSeconds > time_limit)
						{
							// time limit exceeded
							Console.WriteLine($"NMS time limit {time_limit}s exceeded");
						}
						output[xi] = output[xi].MoveToOuterDisposeScope();

					}

					return output;
				}
			}

		}
	}
}
