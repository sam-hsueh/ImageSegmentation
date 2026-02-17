using System;
using System.Collections.Generic;
using System.IO;

namespace Utilities
{
    internal class CsvIO
    {
        public static bool SaveData(int index, float[] x, float[] z, string FilePath = null)
        {
            if (x == null || x.Length <= 20 || z == null)
                return false;
            try
            {
                if (FilePath == null)
                    FilePath = AppDomain.CurrentDomain.BaseDirectory;
                string format = "yyyy-MM-dd";
                string filePath = DateTime.Now.ToString(format, System.Globalization.CultureInfo.CurrentCulture);
                format = "HH-mm-ss";
                FilePath = Path.Combine(FilePath, filePath);
                string fileName = DateTime.Now.ToString(format, System.Globalization.CultureInfo.CurrentCulture);
                string fPath = Path.Combine(FilePath, fileName + "-" + index + ".csv");
                if (!Directory.Exists(FilePath))
                    Directory.CreateDirectory(FilePath);
                using (StreamWriter writer = new StreamWriter(fPath, true))
                {
                    for (int i = 0; i < x.Length; i++)
                    {
                        writer.WriteLine(x[i] + "," + z[i]);
                    }
                    writer.Close();
                }
                return true;
            }
            catch (Exception e)
            {
                return false;
            }
        }
        public static void ReadCSV(string FilePath, out float[] x, out float[] z)
        {
            var tx = new List<string>();
            var tz = new List<string>();
            using (var readCsv = new StreamReader(FilePath))
            {
                while (!readCsv!.EndOfStream)
                {
                    string? line = readCsv?.ReadLine();
                    string[] record = line!.Split(',');
                    tx.Add(record[0]);
                    tz.Add(record[1]);
                }
            }
            x = Array.ConvertAll<string, float>(tx.ToArray(), s => float.Parse(s));
            z = Array.ConvertAll<string, float>(tz.ToArray(), s => float.Parse(s));
        }
    }
}
