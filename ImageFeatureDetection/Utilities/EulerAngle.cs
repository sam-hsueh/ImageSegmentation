using System;

namespace Utilities
{
    internal class EulerAngle
    {
        //X_1-Y_2-Z_3=[■(C2C3           &-C2S3          &S2
        //               @C1S3+C3S1S2    &C1C3-S1S2S3    &-C2S1
        //               @S1S3-C1C3S2    &C3S1+C1S2S3    &C1C2)]
        public static double[] EulerX1Y2Z3(double[] n, double[] o, double[] a)
        {
            double t1 = Math.Atan2(-a[1], a[2]);
            double s1 = Math.Sin(t1);
            double c1 = Math.Cos(t1);
            double t3 = Math.Atan2(c1 * n[1] + s1 * n[2], c1 * o[1] + s1 * o[2]);
            //double t3 = Math.Atan2(-o[0], n[0]);
            double s3 = Math.Sin(t3);
            double c3 = Math.Cos(t3);
            double t2 = Math.Atan2(a[0], c3 * n[0] - s3 * n[1]);
            return new double[] { t1 * 57.29578, t2 * 57.29578, t3 * 57.29578 };
        }
        //X_1-Y_2-X_3=[■(C2       &S2S3           &C3S2
        //               @S1S2     &C1C3-C2S1S3    &-C1S3-C2C3S1
        //               @-C1S2    &C3S1+C1C2S3    &C1C2C3-S1S3)]
        public static double[] EulerX1Y2X3(double[] n, double[] o, double[] a)
        {
            double t1 = Math.Atan2(n[1], -n[2]);
            double s1 = Math.Sin(t1);
            double c1 = Math.Cos(t1);
            double t3 = Math.Atan2(-c1 * a[1] - s1 * a[2], c1 * o[1] + s1 * o[2]);
            //double t3 = Math.Atan2(-o[0], n[0]);
            double s3 = Math.Sin(t3);
            double c3 = Math.Cos(t3);
            double t2 = Math.Atan2(s1 * n[1] - c1 * n[2], n[0]);
            return new double[] { t1 * 57.29578, t2 * 57.29578, t3 * 57.29578 };
        }
        //X_1-Z_2-X_3=[■(C2      &-C3S2          &S2S3
        //               @C1S2    &-S1S3+C1C2C3   &-C3S1-C1C2S3
        //               @S1S2    &C1S3+C2C3S1    &C1C3-C2S1S3)]
        public static double[] EulerX1Z2X3(double[] n, double[] o, double[] a)
        {
            double t1 = Math.Atan2(n[2], n[1]);
            double s1 = Math.Sin(t1);
            double c1 = Math.Cos(t1);
            double t3 = Math.Atan2(c1 * o[2] - s1 * o[1], c1 * a[2] - s1 * a[1]);
            //double t3 = Math.Atan2(-o[0], n[0]);
            double s3 = Math.Sin(t3);
            double c3 = Math.Cos(t3);
            double t2 = Math.Atan2(c1 * n[1] + s1 * n[2], n[0]);
            return new double[] { t1 * 57.29578, t2 * 57.29578, t3 * 57.29578 };
        }
        //X_1-Z_2-Y_3=[■(C2C3           &-S2     &C2S3
        //               @S1S3+C1C3S2    &C1C2    &-C3S1+C1S2S3
        //               @-C1S3+C3S1S2   &S1C2    &C1C3+S1S2S3)]
        public static double[] EulerX1Z2Y3(double[] n, double[] o, double[] a)
        {
            double t1 = Math.Atan2(o[2], o[1]);
            double s1 = Math.Sin(t1);
            double c1 = Math.Cos(t1);
            double t3 = Math.Atan2(s1 * n[1] - c1 * n[2], c1 * a[2] - s1 * a[1]);
            //double t3 = Math.Atan2(-o[0], n[0]);
            double s3 = Math.Sin(t3);
            double c3 = Math.Cos(t3);
            double t2 = Math.Atan2(c1 * o[1] + s1 * o[2], o[0]);
            return new double[] { t1 * 57.29578, t2 * 57.29578, t3 * 57.29578 };
        }

        //Y_1-Z_2-X_3=[■(C1C2    &S1S3-C1C3S2    &C3S1+C1S2S3
        //               @S2      &C2C3           &-C2S3
        //               @-S1C2   &C1S3+C3S1S2    &C1C3-S1S2S3)]
        public static double[] EulerY1Z2X3(double[] n, double[] o, double[] a)
        {
            double t1 = Math.Atan2(-n[2], n[0]);
            double s1 = Math.Sin(t1);
            double c1 = Math.Cos(t1);
            double t3 = Math.Atan2(c1 * o[2] + s1 * o[0], c1 * a[2] + s1 * a[0]);
            //double t3 = Math.Atan2(-a[1], o[1]);
            //double s3 = Math.Sin(t3);
            //double c3 = Math.Cos(t3);
            double t2 = Math.Atan2(n[1], c1 * n[0] - s1 * n[2]);
            return new double[] { t1 * 57.29578, t2 * 57.29578, t3 * 57.29578 };
        }
        //Y_1-Z_2-Y_3=[■(-S1S3+C1C2C3    &-C1S2    &C3S1+C1C2S3
        //               @C3S2           &C2       &S2S3
        //               @-C1S3-C2C3S1   &S1S2     &C1C3-C2S1S3)]
        public static double[] EulerY1Z2Y3(double[] n, double[] o, double[] a)
        {
            double t1 = Math.Atan2(o[2], -o[0]);
            double s1 = Math.Sin(t1);
            double c1 = Math.Cos(t1);
            double t3 = Math.Atan2(-c1 * n[2] - s1 * n[0], c1 * a[2] + s1 * a[0]);
            //double t3 = Math.Atan2(a[1], n[1]);
            //double s3 = Math.Sin(t3);
            //double c3 = Math.Cos(t3);
            double t2 = Math.Atan2(s1 * o[2] - c1 * o[0], o[1]);
            return new double[] { t1 * 57.29578, t2 * 57.29578, t3 * 57.29578 };
        }
        //Y_1-X_2-Y_3=[■(C1C3-C2S1S3    &S1S2    &C1S3+C2C3S1
        //               @S2S3           &C2      &-C3S2
        //               @-C3S1-C1C2S3   &C1S2    &-S1S3+C1C2C3)]
        public static double[] EulerY1X2Y3(double[] n, double[] o, double[] a)
        {
            double t1 = Math.Atan2(o[0], o[2]);
            double s1 = Math.Sin(t1);
            double c1 = Math.Cos(t1);
            double t3 = Math.Atan2(c1 * a[0] - s1 * a[2], c1 * n[0] - s1 * n[2]);
            //double t3 = Math.Atan2(n[1], -a[1]);
            //double s3 = Math.Sin(t3);
            //double c3 = Math.Cos(t3);
            double t2 = Math.Atan2(s1 * o[0] + c1 * o[2], o[1]);
            return new double[] { t1 * 57.29578, t2 * 57.29578, t3 * 57.29578 };
        }
        //Y_1-X_2-Z_3=[■(C1C3+S1S2S3    &C3S1S2-C1S3    &C2S1
        //               @C2S3           &C2C3           &-S2
        //               @-C3S1+C1S2S3    &C1C3S2+S1S3    &C2C1)]
        public static double[] EulerY1X2Z3(double[] n, double[] o, double[] a)
        {
            double t1 = Math.Atan2(a[0], a[2]);
            double s1 = Math.Sin(t1);
            double c1 = Math.Cos(t1);
            double t3 = Math.Atan2(s1 * o[1] - c1 * o[0], -c1 * n[2] + s1 * n[0]);
            //double t3 = Math.Atan2(n[1], o[1]);
            //double s3 = Math.Sin(t3);
            //double c3 = Math.Cos(t3);
            double t2 = Math.Atan2(-a[1], c1 * a[2] + s1 * a[1]);
            return new double[] { t1 * 57.29578, t2 * 57.29578, t3 * 57.29578 };
        }

        //Z_1-X_2-Y_3=[■(C1C3-S1S2S3    &-C2S1    &C1S3+C3S1S2
        //               @S1C3+C1S2S3    &C2C1     &S1S3-C1C2S2
        //               @-C2S3          &S2       &C2C3)]
        public static double[] EulerZ1X2Y3(double[] n, double[] o, double[] a)
        {
            double t1 = Math.Atan2(-o[0], o[1]);
            double s1 = Math.Sin(t1);
            double c1 = Math.Cos(t1);
            double t3 = Math.Atan2(c1 * a[0] + s1 * a[1], c1 * n[0] + s1 * n[1]);
            //double t3 = Math.Atan2(-n[2], a[2]);
            //double s3 = Math.Sin(t3);
            //double c3 = Math.Cos(t3);
            double t2 = Math.Atan2(o[2], c1 * o[1] - s1 * o[0]);
            return new double[] { t1 * 57.29578, t2 * 57.29578, t3 * 57.29578 };
        }
        //Z_1-X_2-Z_3=[■(C1C3-C2S1S3    &-C1S3-C2C3S1    &S1S2
        //               @S1C3+C1C2S3    &-S1S3+C1C2C3    &-C1S2
        //               @S2S3           &S2C3            &C2)]
        public static double[] EulerZ1X2Z3(double[] n, double[] o, double[] a)
        {
            double t1 = Math.Atan2(n[0], -n[1]);
            double s1 = Math.Sin(t1);
            double c1 = Math.Cos(t1);
            double t3 = Math.Atan2(-c1 * o[0] - s1 * o[1], c1 * n[0] + s1 * n[1]);
            //double t3 = Math.Atan2(n[2], o[2]);
            //double s3 = Math.Sin(t3);
            //double c3 = Math.Cos(t3);
            double t2 = Math.Atan2(s1 * a[0] - c1 * a[1], a[2]);
            return new double[] { t1 * 57.29578, t2 * 57.29578, t3 * 57.29578 };
        }
        //Z_1-Y_2-X_3=[■(C1C2    &C1S2S3-C3S1    &S1S3+C1C3S2
        //               @S1C2    &C1C3+S1S2S3    &C3S1S2-C1S3
        //               @-S2     &C2S3           &C2C3)]
        public static double[] EulerZ1Y2X3(double[] n, double[] o, double[] a)//Fanuc WPR
        {
            double t1, t2, t3;
            double c2 = Math.Sqrt(n[0] * n[0] + n[1] * n[1]);
            if (Math.Abs(c2) < 1e-6)
            {
                if (double.IsPositiveInfinity(-n[2] / c2))
                {
                    t1 = 0;
                    t2 = Math.PI / 2;
                    t3 = Math.Atan2(o[0], o[1]);
                }
                else if (double.IsNegativeInfinity(-n[2] / c2))
                {
                    t1 = 0;
                    t2 = -Math.PI / 2;
                    t3 = -Math.Atan2(o[0], o[1]);
                }
                else
                {
                    t1 = Math.Atan2(-a[1], o[1]);
                    t2 = Math.Atan2(-n[2], c2);
                    t3 = 0;
                }
            }
            else
            {
                t2 = Math.Atan2(-n[2], c2);
                t1 = Math.Atan2(o[2], a[2]);
                t3 = Math.Atan2(n[1], n[0]);
            }
            var tht = t1 * 57.29578;
            if (tht > 0)
                tht -= 180;
            else if (tht < 0)
                tht += 180;
            return new double[] { tht, t2 * 57.29578, t3 * 57.29578 };
        }
        //Z_1-Y_2-X_3=[■(C1C2    &C1S2S3-C3S1    &C1C3S2+S1S3
        //               @S1C2    &S1S2S3+C3C1    &C3S1S2-C1S3
        //               @-S2     &C2S3           &C2C3)]
        public static double[] EulerZ1Y2X32(double[] n, double[] o, double[] a)
        {
            double t1 = Math.Atan2(n[1], n[0]);
            double s1 = Math.Sin(t1);
            double c1 = Math.Cos(t1);
            double t3 = Math.Atan2(c1 * o[1] - s1 * o[0], -c1 * a[1] + s1 * a[0]);
            //double t3 = Math.Atan2(o[2], n[2]);
            //double s3 = Math.Sin(t3);
            //double c3 = Math.Cos(t3);
            double t2 = Math.Atan2(-n[2], c1 * n[0] + s1 * n[1]);
            return new double[] { t1 * 57.29578, t2 * 57.29578, t3 * 57.29578 };
        }
        //Z_1-Y_2-Z_3=[■(C1C2C3-S1S3  &-S1C3-C1C2S3  &C1S2
        //               @C2C3S1+C1S3  &C1C3-C2S1S3   &S1S2
        //               @-C3S2        &S2S3          &C2)]
        public static double[] EulerZ1Y2Z3(double[] n, double[] o, double[] a)
        {
            double t1 = Math.Atan2(a[1], a[0]);
            double s1 = Math.Sin(t1);
            double c1 = Math.Cos(t1);
            double t3 = Math.Atan2(c1 * n[1] - s1 * n[0], c1 * o[1] - s1 * o[0]);
            //double t3 = Math.Atan2(o[2], -n[2]);
            //double s3 = Math.Sin(t3);
            //double c3 = Math.Cos(t3);
            double t2 = Math.Atan2(c1 * a[0] + s1 * a[1], a[2]);
            return new double[] { t1 * 57.29578, t2 * 57.29578, t3 * 57.29578 };
        }
    }
}
