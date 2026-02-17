using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Utilities
{
    public static class Extensions
    {
        static double Phi(double x)
        {
            // constants
            double a1 = 0.254829592;
            double a2 = -0.284496736;
            double a3 = 1.421413741;
            double a4 = -1.453152027;
            double a5 = 1.061405429;
            double p = 0.3275911;

            // Save the sign of x
            int sign = 1;
            if (x < 0)
                sign = -1;
            x = Math.Abs(x) / Math.Sqrt(2.0);

            // A&S formula 7.1.26
            double t = 1.0 / (1.0 + p * x);
            double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

            return 0.5 * (1.0 + sign * y);
        }

        static void TestPhi()
        {
            // Select a few input values
            double[] x = { -3, -1, 0.0, 0.5, 2.1 };

            // Output computed by Mathematica
            // y = Phi[x]
            double[] y = { 0.00134989803163, 0.158655253931, 0.5, 0.691462461274, 0.982135579437 };

            double maxError = 0.0;
            for (int i = 0; i < x.Length; ++i)
            {
                double error = Math.Abs(y[i] - Phi(x[i]));
                if (error > maxError)
                    maxError = error;
            }

            Console.WriteLine("Maximum error: {0}", maxError);
        }
        /// <summary>
        /// 获取一个指定范围内的标准正态分布随机数
        /// </summary>
        /// <param name="min">最小值（返回的值可包含最小值）</param>
        /// <param name="max">最大值（返回的值可包含最大值）</param>
        /// <returns></returns>
        public static double Random(double min, double max) => Random2(min, max)[0];
        public const int MIU = 0;
        public const int SIGMA = 1;

        /// <summary>
        /// 获取一个或两个指定范围内的标准正态分布随机数
        /// </summary>
        /// <param name="min">最小值（返回的值可包含最小值）</param>
        /// <param name="max">最大值（返回的值可包含最大值）</param>
        /// <returns></returns>
        public static double[] Random2(double min, double max)
        {
            if (max <= min) return new[] { min };
            Random rnd = new Random();
            var list = new List<double>();
            while (list.Count == 0)
            {
                foreach (var normal in GetValue(rnd, SIGMA, MIU))
                {
                    var value = ConvertValue(normal, min, max);
                    if (value >= min && value <= max)
                    {
                        list.Add(value);
                    }
                }
            }

            return list.ToArray();
        }

        /// <summary>
        /// 转换为指定范围内的值。返回结果可能会超出范围为，因为大约99.7％概率在范围内（3σ原则）。
        /// </summary>
        /// <param name="normal">正态分布数</param>
        /// <param name="min">最小值</param>
        /// <param name="max">最大值</param>
        /// <returns></returns>
        private static double ConvertValue(double normal, double min, double max)
        {
            var median = (max - min) / 2.0;

            return (normal * median / 3) + min + median;
        }

        /// <summary>
        /// 获取两个正态分布随机数
        /// </summary>
        /// <param name="random">生成随机数的对象</param>
        /// <param name="deviation">标准差σ</param>
        /// <param name="expected">期望值μ</param>
        /// <returns></returns>
        public static double[] GetValue(Random random, double deviation, double expected)
        {
            double x, y, s;
            do
            {
                x = 2 * random.NextDouble() - 1;
                y = 2 * random.NextDouble() - 1;
                s = x * x + y * y;
            }
            while (s > 1 || s == 0);
            double multi = Math.Sqrt(-2 * Math.Log(s) / s);

            //return new double[] { Math.Round(x * multi, 2) * deviation + expected, Math.Round(y * multi, 2) * deviation + expected };
            return new double[] { x * multi * deviation + expected, y * multi * deviation + expected };
        }
    }
}
