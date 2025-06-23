//using System;
//using System.Collections.Generic;
//using System.Drawing;
//using System.Formats.Asn1;
//using System.IO;
//using System.Linq;
//using System.Net.Http.Json;
//using System.Text;
//using System.Text.Json;
//using System.Text.Json.Serialization;
//using System.Threading.Tasks;

//namespace ImageSegmentation.Utilities
//{
//    [Serializable]
//    public class CBase
//    {

//        [JsonConverter(typeof(ImageConverter))]
//        public Image Img
//        { get; set; }
//    }
//    public class ImageConverter : JsonConverter
//    {
//        public override bool CanConvert(Type objectType)
//        {
//            return objectType == typeof(Bitmap);
//        }

//        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
//        {
//            var m = new MemoryStream(Convert.FromBase64String((string)reader.Value));
//            return (Bitmap)Bitmap.FromStream(m);
//        }

//        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
//        {
//            Bitmap bmp = (Bitmap)value;
//            MemoryStream m = new MemoryStream();
//            bmp.Save(m, System.Drawing.Imaging.ImageFormat.Jpeg);

//            writer.WriteValue(Convert.ToBase64String(m.ToArray()));
//        }
//        //把对象序列化
//        string jsonStr = JsonConvert.SerializeObject(obj);  //序列化对象
//        public static T ParseJson<T>(string jsonstr)
//        {
//            return JsonConvert.DeserializeObject<T>(jsonstr);
//        }

//        //直接调用,反序列化到对象
//        CBase b = ParseJson<CBase>(jsonstr);
//    }
//}
