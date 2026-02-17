using System;
using System.Collections.Generic;
using System.Linq;

namespace Utilities
{
    public static class DoubleArrayExtensions
    {
        public static double[] CholeskyDecompositionInverse3(double[,] source, double[] V, double lambda = 0.0003)
        {
            double A11 = source[0, 0] + lambda;
            double A22 = source[1, 1] + lambda;
            double A33 = source[2, 2] + lambda;
            //         Cholesly decomposition
            double G11 = Math.Sqrt(A11);
            double G12 = source[0, 1] / G11;
            double G13 = source[0, 2] / G11;

            double G22 = Math.Sqrt(A22 - G12 * G12);
            double G23 = (source[1, 2] - G12 * G13) / G22;

            double G33 = Math.Sqrt(A33 - G13 * G13 - G23 * G23);

            double D1 = V[0] / G11;
            double D2 = (V[1] - G12 * D1) / G22;
            double D3 = (V[2] - G13 * D1 - G23 * D2) / G33;

            double R1 = D3 / G33;
            double R2 = (D2 - G23 * R1) / G22;
            double R3 = (D1 - G12 * R2 - G13 * R1) / G11;
            return new double[] { R3, R2, R1 };
        }
        public static double[] CholeskyDecompositionInverse4(double[,] source, double[] V)
        {
            source[0, 0] = Math.Sqrt(source[0, 0]);
            source[1, 0] = source[0, 1] / source[0, 0];
            source[2, 0] = source[0, 2] / source[0, 0];
            source[3, 0] = source[0, 3] / source[0, 0];

            double G22 = Math.Sqrt(source[1, 1] - source[1, 0] * source[1, 0]);
            double G23 = (source[1, 2] - source[1, 0] * source[2, 0]) / G22;
            double G24 = (source[1, 3] - source[1, 0] * source[3, 0]) / G22;

            double G33 = Math.Sqrt(source[2, 2] - source[2, 0] * source[2, 0] - G23 * G23);
            double G34 = (source[2, 3] - source[2, 0] * source[3, 0] - G23 * G24) / G33;

            double G44 = Math.Sqrt(source[3, 3] - source[3, 0] * source[3, 0] - G24 * G24 - G34 * G34);

            double D1 = V[0] / source[0, 0];
            double D2 = (V[1] - source[1, 0] * D1) / G22;
            double D3 = (V[2] - source[2, 0] * D1 - G23 * D2) / G33;
            double D4 = (V[3] - source[3, 0] * D1 - G24 * D2 - G34 * D3) / G44;

            V[3] = D4 / G44;
            V[2] = (D3 - G34 * V[3]) / G33;
            V[1] = (D2 - G23 * V[2] - G24 * V[3]) / G22;
            V[0] = (D1 - source[1, 0] * V[1] - source[2, 0] * V[2] - source[3, 0] * V[3]) / source[0, 0];
            return V;
        }

        public static double[] CholeskyDecompositionInverse40(double[,] source, double[] V)
        {
            double G11 = Math.Sqrt(source[0, 0]);
            double G12 = source[0, 1] / G11;
            double G13 = source[0, 2] / G11;
            double G14 = source[0, 3] / G11;

            double G22 = Math.Sqrt(source[1, 1] - G12 * G12);
            double G23 = (source[1, 2] - G12 * G13) / G22;
            double G24 = (source[1, 3] - G12 * G14) / G22;

            double G33 = Math.Sqrt(source[2, 2] - G13 * G13 - G23 * G23);
            double G34 = (source[2, 3] - G13 * G14 - G23 * G24) / G33;

            double G44 = Math.Sqrt(source[3, 3] - G14 * G14 - G24 * G24 - G34 * G34);

            double D1 = V[0] / G11;
            double D2 = (V[1] - G12 * D1) / G22;
            double D3 = (V[2] - G13 * D1 - G23 * D2) / G33;
            double D4 = (V[3] - G14 * D1 - G24 * D2 - G34 * D3) / G44;

            V[3] = D4 / G44;
            V[2] = (D3 - G34 * V[3]) / G33;
            V[1] = (D2 - G23 * V[2] - G24 * V[3]) / G22;
            V[0] = (D1 - G12 * V[1] - G13 * V[2] - G14 * V[3]) / G11;
            return V;
        }
        public static double[] CholeskyDecompositionInverse4Unsafe(double[,] source, double[] V, double[] b = null)
        {
            unsafe
            {
                fixed (double* a = source)
                fixed (double* v = V)
                {
                    double* ca = a;
                    double G11 = Math.Sqrt(ca[0]);
                    double G12 = ca[1] / G11;
                    double G13 = ca[2] / G11;
                    double G14 = ca[3] / G11;
                    ca += 4;
                    double G22 = Math.Sqrt(ca[1] - G12 * G12);
                    double G23 = (ca[2] - G12 * G13) / G22;
                    double G24 = (ca[3] - G12 * G14) / G22;
                    ca += 4;
                    double G33 = Math.Sqrt(ca[2] - G13 * G13 - G23 * G23);
                    double G34 = (ca[3] - G13 * G14 - G23 * G24) / G33;
                    ca += 4;
                    double G44 = Math.Sqrt(ca[3] - G14 * G14 - G24 * G24 - G34 * G34);


                    v[0] = v[0] / G11;
                    v[1] = (v[1] - G12 * v[0]) / G22;
                    v[2] = (v[2] - G13 * v[0] - G23 * v[1]) / G33;
                    v[3] = (v[3] - G14 * v[0] - G24 * v[1] - G34 * v[2]) / G44;

                    v[3] = v[3] / G44;
                    v[2] = (v[2] - G34 * v[3]) / G33;
                    v[1] = (v[1] - G23 * v[2] - G24 * v[3]) / G22;
                    v[0] = (v[0] - G12 * v[1] - G13 * v[2] - G14 * v[3]) / G11;
                    if (b != null)
                        fixed (double* pb = b)
                        {
                            v[0] += pb[0];
                            v[1] += pb[1];
                            v[2] += pb[2];
                            v[3] += pb[3];
                        }
                }
            }
            return V;
        }
        public static double[] CholeskyDecompositionInverse40(double[,] source, double[] V, double[] b = null)
        {
            source[0, 0] = Math.Sqrt(source[0, 0]);
            source[1, 0] = source[0, 1] / source[0, 0];
            source[2, 0] = source[0, 2] / source[0, 0];
            source[3, 0] = source[0, 3] / source[0, 0];

            source[1, 1] = Math.Sqrt(source[1, 1] - source[1, 0] * source[1, 0]);
            source[2, 1] = (source[1, 2] - source[1, 0] * source[2, 0]) / source[1, 1];
            source[3, 1] = (source[1, 3] - source[1, 0] * source[3, 0]) / source[1, 1];

            source[2, 2] = Math.Sqrt(source[2, 2] - source[2, 0] * source[2, 0] - source[2, 1] * source[2, 1]);
            source[3, 2] = (source[2, 3] - source[2, 0] * source[3, 0] - source[2, 1] * source[3, 1]) / source[2, 2];

            source[3, 3] = Math.Sqrt(source[3, 3] - source[3, 0] * source[3, 0] - source[3, 1] * source[3, 1] - source[3, 2] * source[3, 2]);

            V[0] = V[0] / source[0, 0];
            V[1] = (V[1] - source[1, 0] * V[0]) / source[1, 1];
            V[2] = (V[2] - source[2, 0] * V[0] - source[2, 1] * V[1]) / source[2, 2];
            V[3] = (V[3] - source[3, 0] * V[0] - source[3, 1] * V[1] - source[3, 2] * V[2]) / source[3, 3];

            V[3] = V[3] / source[3, 3];
            V[2] = (V[2] - source[3, 2] * V[3]) / source[2, 2];
            V[1] = (V[1] - source[2, 1] * V[2] - source[3, 1] * V[3]) / source[1, 1];
            V[0] = (V[0] - source[1, 0] * V[1] - source[2, 0] * V[2] - source[3, 0] * V[3]) / source[0, 0];
            if (b != null)
            {
                V[0] += b[0];
                V[1] += b[1];
                V[2] += b[2];
                V[3] += b[3];
            }
            return V;
        }
        public static double[] CholeskyDecompositionInverse4(double[,] source, double[] V, double[] b = null)
        {
            double G11 = Math.Sqrt(source[0, 0]);
            double G12 = source[0, 1] / G11;
            double G13 = source[0, 2] / G11;
            double G14 = source[0, 3] / G11;

            double G22 = Math.Sqrt(source[1, 1] - G12 * G12);
            double G23 = (source[1, 2] - G12 * G13) / G22;
            double G24 = (source[1, 3] - G12 * G14) / G22;

            double G33 = Math.Sqrt(source[2, 2] - G13 * G13 - G23 * G23);
            double G34 = (source[2, 3] - G13 * G14 - G23 * G24) / G33;

            double G44 = Math.Sqrt(source[3, 3] - G14 * G14 - G24 * G24 - G34 * G34);

            V[0] = V[0] / G11;
            V[1] = (V[1] - G12 * V[0]) / G22;
            V[2] = (V[2] - G13 * V[0] - G23 * V[1]) / G33;
            V[3] = (V[3] - G14 * V[0] - G24 * V[1] - G34 * V[2]) / G44;

            V[3] = V[3] / G44;
            V[2] = (V[2] - G34 * V[3]) / G33;
            V[1] = (V[1] - G23 * V[2] - G24 * V[3]) / G22;
            V[0] = (V[0] - G12 * V[1] - G13 * V[2] - G14 * V[3]) / G11;
            if (b != null)
            {
                V[0] += b[0];
                V[1] += b[1];
                V[2] += b[2];
                V[3] += b[3];
            }
            return V;
        }


        public static double[] CholeskyDecompositionInverse5Unsafe(double[,] source, double[] V, double[] b = null)
        {
            unsafe
            {
                fixed (double* a = source)
                fixed (double* v = V)
                {
                    double* ca = a;
                    double G11 = Math.Sqrt(ca[0]);
                    double G12 = ca[1] / G11;
                    double G13 = ca[2] / G11;
                    double G14 = ca[3] / G11;
                    double G15 = ca[4] / G11;
                    ca += 5;
                    double G22 = Math.Sqrt(ca[1] - G12 * G12);
                    double G23 = (ca[2] - G12 * G13) / G22;
                    double G24 = (ca[3] - G12 * G14) / G22;
                    double G25 = (ca[4] - G12 * G15) / G22;
                    ca += 5;
                    double G33 = Math.Sqrt(ca[2] - G13 * G13 - G23 * G23);
                    double G34 = (ca[3] - G13 * G14 - G23 * G24) / G33;
                    double G35 = (ca[4] - G13 * G15 - G23 * G25) / G33;
                    ca += 5;
                    double G44 = Math.Sqrt(ca[3] - G14 * G14 - G24 * G24 - G34 * G34);
                    double G45 = (ca[4] - G14 * G15 - G24 * G25 - G34 * G35) / G44;
                    ca += 5;
                    double G55 = Math.Sqrt(ca[4] - G15 * G15 - G25 * G25 - G35 * G35 - G45 * G45);

                    v[0] = v[0] / G11;
                    v[1] = (v[1] - G12 * v[0]) / G22;
                    v[2] = (v[2] - G13 * v[0] - G23 * v[1]) / G33;
                    v[3] = (v[3] - G14 * v[0] - G24 * v[1] - G34 * v[2]) / G44;
                    v[4] = (v[4] - G15 * v[0] - G25 * v[1] - G35 * v[2] - G45 * v[3]) / G55;

                    v[4] = v[4] / G55;
                    v[3] = (v[3] - G45 * v[4]) / G44;
                    v[2] = (v[2] - G34 * v[3] - G35 * v[4]) / G33;
                    v[1] = (v[1] - G23 * v[2] - G24 * v[3] - G25 * v[4]) / G22;
                    v[0] = (v[0] - G12 * v[1] - G13 * v[2] - G14 * v[3] - G15 * v[4]) / G11;
                    if (b != null)
                    {
                        v[0] += b[0];
                        v[1] += b[1];
                        v[2] += b[2];
                        v[3] += b[3];
                        v[4] += b[4];
                    }
                }
            }
            return V;
        }
        public static double[] CholeskyDecompositionInverse50(double[,] source, double[] V, double[] b = null)
        {
            source[0, 0] = Math.Sqrt(source[0, 0]);
            source[1, 0] = source[0, 1] / source[0, 0];
            source[2, 0] = source[0, 2] / source[0, 0];
            source[3, 0] = source[0, 3] / source[0, 0];
            source[4, 0] = source[0, 4] / source[0, 0];

            source[1, 1] = Math.Sqrt(source[1, 1] - source[1, 0] * source[1, 0]);
            source[2, 1] = (source[1, 2] - source[1, 0] * source[2, 0]) / source[1, 1];
            source[3, 1] = (source[1, 3] - source[1, 0] * source[3, 0]) / source[1, 1];
            source[4, 1] = (source[1, 4] - source[1, 0] * source[4, 0]) / source[1, 1];

            source[2, 2] = Math.Sqrt(source[2, 2] - source[2, 0] * source[2, 0] - source[2, 1] * source[2, 1]);
            source[3, 2] = (source[2, 3] - source[2, 0] * source[3, 0] - source[2, 1] * source[3, 1]) / source[2, 2];
            source[4, 2] = (source[2, 4] - source[2, 0] * source[4, 0] - source[2, 1] * source[4, 1]) / source[2, 2];

            source[3, 3] = Math.Sqrt(source[3, 3] - source[3, 0] * source[3, 0] - source[3, 1] * source[3, 1] - source[3, 2] * source[3, 2]);
            source[4, 3] = (source[3, 4] - source[3, 0] * source[4, 0] - source[3, 1] * source[4, 1] - source[3, 2] * source[4, 2]) / source[3, 3];

            source[4, 4] = Math.Sqrt(source[4, 4] - source[4, 0] * source[4, 0] - source[4, 1] * source[4, 1] - source[4, 2] * source[4, 2] - source[4, 3] * source[4, 3]);
            V[0] = V[0] / source[0, 0];
            V[1] = (V[1] - source[1, 0] * V[0]) / source[1, 1];
            V[2] = (V[2] - source[2, 0] * V[0] - source[2, 1] * V[1]) / source[2, 2];
            V[3] = (V[3] - source[3, 0] * V[0] - source[3, 1] * V[1] - source[3, 2] * V[2]) / source[3, 3];
            V[4] = (V[4] - source[4, 0] * V[0] - source[4, 1] * V[1] - source[4, 2] * V[2] - source[4, 3] * V[3]) / source[4, 4];

            V[4] = V[4] / source[4, 4];
            V[3] = (V[3] - source[4, 3] * V[4]) / source[3, 3];
            V[2] = (V[2] - source[3, 2] * V[3] - source[4, 2] * V[4]) / source[2, 2];
            V[1] = (V[1] - source[2, 1] * V[2] - source[3, 1] * V[3] - source[4, 1] * V[4]) / source[1, 1];
            V[0] = (V[0] - source[1, 0] * V[1] - source[2, 0] * V[2] - source[3, 0] * V[3] - source[4, 0] * V[4]) / source[0, 0];
            if (b != null)
            {
                V[0] += b[0];
                V[1] += b[1];
                V[2] += b[2];
                V[3] += b[3];
                V[4] += b[4];
            }
            return V;
        }

        public static double[] CholeskyDecompositionInverse5(double[,] source, double[] V, double[] b = null)
        {
            double G11 = Math.Sqrt(source[0, 0]);
            double G12 = source[0, 1] / G11;
            double G13 = source[0, 2] / G11;
            double G14 = source[0, 3] / G11;
            double G15 = source[0, 4] / G11;

            double G22 = Math.Sqrt(source[1, 1] - G12 * G12);
            double G23 = (source[1, 2] - G12 * G13) / G22;
            double G24 = (source[1, 3] - G12 * G14) / G22;
            double G25 = (source[1, 4] - G12 * G15) / G22;

            double G33 = Math.Sqrt(source[2, 2] - G13 * G13 - G23 * G23);
            double G34 = (source[2, 3] - G13 * G14 - G23 * G24) / G33;
            double G35 = (source[2, 4] - G13 * G15 - G23 * G25) / G33;

            double G44 = Math.Sqrt(source[3, 3] - G14 * G14 - G24 * G24 - G34 * G34);
            double G45 = (source[3, 4] - G14 * G15 - G24 * G25 - G34 * G35) / G44;

            double G55 = Math.Sqrt(source[4, 4] - G15 * G15 - G25 * G25 - G35 * G35 - G45 * G45);

            V[0] = V[0] / G11;
            V[1] = (V[1] - G12 * V[0]) / G22;
            V[2] = (V[2] - G13 * V[0] - G23 * V[1]) / G33;
            V[3] = (V[3] - G14 * V[0] - G24 * V[1] - G34 * V[2]) / G44;
            V[4] = (V[4] - G15 * V[0] - G25 * V[1] - G35 * V[2] - G45 * V[3]) / G55;

            V[4] = V[4] / G55;
            V[3] = (V[3] - G45 * V[4]) / G44;
            V[2] = (V[2] - G34 * V[3] - G35 * V[4]) / G33;
            V[1] = (V[1] - G23 * V[2] - G24 * V[3] - G25 * V[4]) / G22;
            V[0] = (V[0] - G12 * V[1] - G13 * V[2] - G14 * V[3] - G15 * V[4]) / G11;
            if (b != null)
            {
                V[0] += b[0];
                V[1] += b[1];
                V[2] += b[2];
                V[3] += b[3];
                V[4] += b[4];
            }
            return V;
        }

        public static double[] CholeskyDecompositionInverse5(double[,] source, double[] V, double lambda = 0.03)
        {
            double A11 = source[0, 0] + lambda;
            double A22 = source[1, 1] + lambda;
            double A33 = source[2, 2] + lambda;
            double A44 = source[3, 3] + lambda;
            double A55 = source[4, 4] + lambda;
            //         Cholesly decomposition
            double G11 = Math.Sqrt(A11);
            double G12 = source[0, 1] / G11;
            double G13 = source[0, 2] / G11;
            double G14 = source[0, 3] / G11;
            double G15 = source[0, 4] / G11;

            double G22 = Math.Sqrt(A22 - G12 * G12);
            double G23 = (source[1, 2] - G12 * G13) / G22;
            double G24 = (source[1, 3] - G12 * G14) / G22;
            double G25 = (source[1, 4] - G12 * G15) / G22;

            double G33 = Math.Sqrt(A33 - G13 * G13 - G23 * G23);
            double G34 = (source[2, 3] - G13 * G14 - G23 * G24) / G33;
            double G35 = (source[2, 4] - G13 * G15 - G23 * G25) / G33;

            double G44 = Math.Sqrt(A44 - G14 * G14 - G24 * G24 - G34 * G34);
            double G45 = (source[3, 4] - G14 * G15 - G24 * G25 - G34 * G35) / G44;

            double G55 = Math.Sqrt(A55 - G15 * G15 - G25 * G25 - G35 * G35 - G45 * G45);

            double D1 = V[0] / G11;
            double D2 = (V[1] - G12 * D1) / G22;
            double D3 = (V[2] - G13 * D1 - G23 * D2) / G33;
            double D4 = (V[3] - G14 * D1 - G24 * D2 - G34 * D3) / G44;
            double D5 = (V[4] - G15 * D1 - G25 * D2 - G35 * D3 - G45 * D4) / G55;

            double R1 = D5 / G55;
            double R2 = (D4 - G45 * R1) / G44;
            double R3 = (D3 - G34 * R2 - G35 * R1) / G33;
            double R4 = (D2 - G23 * R3 - G24 * R2 - G25 * R3) / G22;
            double R5 = (D1 - G12 * R3 - G13 * R2 - G14 * R3 - G15 * R4) / G11;

            return new double[] { R1, R2, R3, R3, R4 };
        }
        public static double[] CholeskyDecompositionInverse6(double[,] source, double[] V, double[] b = null)
        {
            double G11 = Math.Sqrt(source[0, 0]);
            double G12 = source[0, 1] / G11;
            double G13 = source[0, 2] / G11;
            double G14 = source[0, 3] / G11;
            double G15 = source[0, 4] / G11;
            double G16 = source[0, 5] / G11;

            double G22 = Math.Sqrt(source[1, 1] - G12 * G12);
            double G23 = (source[1, 2] - G12 * G13) / G22;
            double G24 = (source[1, 3] - G12 * G14) / G22;
            double G25 = (source[1, 4] - G12 * G15) / G22;
            double G26 = (source[1, 5] - G12 * G16) / G22;


            double G33 = Math.Sqrt(source[2, 2] - G13 * G13 - G23 * G23);
            double G34 = (source[2, 3] - G13 * G14 - G23 * G24) / G33;
            double G35 = (source[2, 4] - G13 * G15 - G23 * G25) / G33;
            double G36 = (source[2, 5] - G13 * G16 - G23 * G26) / G33;


            double G44 = Math.Sqrt(source[3, 3] - G14 * G14 - G24 * G24 - G34 * G34);
            double G45 = (source[3, 4] - G14 * G15 - G24 * G25 - G34 * G35) / G44;
            double G46 = (source[3, 5] - G14 * G16 - G24 * G26 - G34 * G36) / G44;

            double G55 = Math.Sqrt(source[4, 4] - G15 * G15 - G25 * G25 - G35 * G35 - G45 * G45);
            double G56 = (source[4, 5] - G15 * G16 - G25 * G26 - G35 * G36 - G45 * G46) / G55;

            double G66 = Math.Sqrt(source[5, 5] - G16 * G16 - G26 * G26 - G36 * G36 - G46 * G46 - G56 * G56);


            V[0] = V[0] / G11;
            V[1] = (V[1] - G12 * V[0]) / G22;
            V[2] = (V[2] - G13 * V[0] - G23 * V[1]) / G33;
            V[3] = (V[3] - G14 * V[0] - G24 * V[1] - G34 * V[2]) / G44;
            V[4] = (V[4] - G15 * V[0] - G25 * V[1] - G35 * V[2] - G45 * V[3]) / G55;
            V[5] = (V[5] - G16 * V[0] - G26 * V[1] - G36 * V[2] - G46 * V[3] - G56 * V[4]) / G66;

            V[5] = V[5] / G66;
            V[4] = (V[4] - G56 * V[5]) / G55;
            V[3] = (V[3] - G45 * V[4] - G46 * V[5]) / G44;
            V[2] = (V[2] - G34 * V[3] - G35 * V[4] - G36 * V[5]) / G33;
            V[1] = (V[1] - G23 * V[2] - G24 * V[3] - G25 * V[4] - G26 * V[5]) / G22;
            V[0] = (V[0] - G12 * V[1] - G13 * V[2] - G14 * V[3] - G15 * V[4] - G16 * V[5]) / G11;

            if (b != null)
            {
                V[0] += b[0];
                V[1] += b[1];
                V[2] += b[2];
                V[3] += b[3];
                V[4] += b[4];
            }
            return V;
        }

        /// <summary>
        /// 构造正定矩阵
        /// </summary>
        /// <param name="lu">创建正定矩阵</param>
        /// <returns></returns>
        public static double[,] LeastSquareUnsafe(double[] a)
        {
            int rows = a.Length;

            // Copy right hand side with pivoting
            var result = new double[rows, rows];

            unsafe
            {
                fixed (double* A = a)
                fixed (double* px = result)
                {
                    double* X = px;
                    for (int i = 0; i < rows; i++)
                    {
                        for (int j = 0; j < rows; j++)
                        {
                            *(X++) = A[i] * A[j];
                        }
                    }
                    /*
                    for (int i = 0; i < rows; i++)
                    {
                        double* X = &px[i * rows];
                        for (int j = 0; j < rows; j++)
                        {
                            X[j] = A[i] * A[j];
                        }
                    }
                     */
                }
            }
            return result;
        }
        /// <summary>
        /// 矩阵相乘
        /// </summary>
        /// <param name="a">左矩阵</param>
        /// <param name="b">右矩阵</param>
        /// <returns></returns>
        public static double[,] MultiplyUnsafe(this double[,] a, double[,] b)
        {
            double[,] result;
            int N = a.GetLength(0);
            int M = b.GetLength(1);
            int K = a.GetLength(1);
            var t = new double[K];
            if (K != b.GetLength(0)) return null;

            result = new double[N, M];
            unsafe
            {
                fixed (double* A = a)
                fixed (double* B = b)
                fixed (double* T = t)
                fixed (double* R = result)
                {
                    for (int j = 0; j < M; j++)
                    {
                        double* pb = B + j;
                        for (int k = 0; k < K; k++)
                        {
                            T[k] = *pb;
                            pb += M;
                        }

                        double* pa = A;
                        double* pr = R + j;
                        for (int i = 0; i < N; i++)
                        {
                            double s = (double)0;
                            for (int k = 0; k < K; k++)
                                s += (double)((double)pa[k] * (double)T[k]);
                            *pr = (double)s;
                            pa += K;
                            pr += M;
                        }
                    }
                }
            }
            t = null;
            return result;
        }
        /// <summary>
        /// 矩阵相乘
        /// </summary>
        /// <param name="a">左矩阵</param>
        /// <param name="b">右矩阵</param>
        /// <returns></returns>
        public static double[,] Multiply(this double[,] a, double[,] b)
        {
            double[,] result;
            int N = a.GetLength(0);
            int M = b.GetLength(1);
            int K = a.GetLength(1);
            var t = new double[K];
            if (K != b.GetLength(0)) return null;

            result = new double[N, M];
            for (int i = 0; i < N; i++)
                for (int j = 0; j < M; j++)
                    for (int k = 0; k < K; k++)
                        result[i, j] += ((double)a[i, k] * (double)b[k, j]);

            return result;
        }
        /// <summary>
        /// 对称正定矩阵求逆
        /// </summary>
        /// <param name="a">求逆矩阵</param>
        /// <returns></returns>
        public static double[,] InvertSsgjUnsafeU(double[,] Mat)
        {
            double[,] a = (double[,])Mat.Clone();
            int n = a.GetLength(0);
            if (a.GetLength(0) != a.GetLength(1)) return null;

            double[] t = new double[n];
            unsafe
            {
                fixed (double* pa = a)
                fixed (double* pt = t)
                {
                    double* ca;
                    double g, w;
                    for (int k = 0; k < n; k++)
                    {
                        w = pa[0];
                        if (w == 0.0)
                            return null;
                        int m = n - k - 1;
                        ca = pa;
                        for (int i = 1; i < n; i++)
                        {
                            g = pa[i];
                            pt[i] = g / w;
                            if (i <= m)
                                pt[i] = -pt[i];
                            for (int j = 1; j <= i; j++)
                                ca[(j - 1) * n + i - 1] = ca[n * j + i] + g * pt[j];
                        }

                        ca[n * n - 1] = 1.0 / w;
                        for (int i = 1; i < n; i++)
                            ca[i * n - 1] = pt[i];
                    }
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < i; j++)
                            pa[i * n + j] = pa[j * n + i];
                }
            }
            return a;
        }
        public static double[,] InvertSsgjUnsafeL(double[,] a)
        {
            int n = a.GetLength(0);
            if (a.GetLength(0) != a.GetLength(1)) return null;
            double[] t = new double[n];
            unsafe
            {
                fixed (double* pa = a)
                fixed (double* pt = t)
                {
                    double* ca;
                    double g, w;
                    for (int k = 0; k < n; k++)
                    {
                        w = pa[0];
                        if (w == 0.0)
                            return null;
                        int m = n - k - 1;
                        ca = pa;
                        for (int i = 1; i < n; i++)
                        {
                            g = pa[i * n];
                            pt[i] = g / w;
                            if (i <= m)
                                pt[i] = -pt[i];
                            for (int j = 1; j <= i; j++)
                                ca[j - 1] = ca[n + j] + g * pt[j];
                            ca += n;
                        }

                        ca[n - 1] = 1.0 / w;
                        for (int i = 1; i < n; i++)
                            ca[i - 1] = pt[i];
                    }

                    for (int i = 0; i < n - 1; i++)
                        for (int j = i + 1; j < n; j++)
                            pa[i * n + j] = pa[j * n + i];
                }
            }
            return a;
        }

        public static double[] InvertSsgjUnsafeU(double[,] a, double[] V)
        {
            int n = a.GetLength(0);
            if (a.GetLength(0) != a.GetLength(1)) return null;

            double[] t = new double[n];
            unsafe
            {
                fixed (double* pa = a)
                fixed (double* pt = t)
                fixed (double* pv = V)
                {
                    double* ca;
                    double g, w;
                    for (int k = 0; k < n; k++)
                    {
                        w = pa[0];
                        if (w == 0.0)
                            return null;
                        int m = n - k - 1;
                        ca = pa;
                        for (int i = 1; i < n; i++)
                        {
                            g = pa[i];
                            pt[i] = g / w;
                            if (i <= m)
                                pt[i] = -pt[i];
                            for (int j = 1; j <= i; j++)
                                ca[(j - 1) * n + i - 1] = ca[n * j + i] + g * pt[j];
                        }

                        ca[n * n - 1] = 1.0 / w;
                        for (int i = 1; i < n; i++)
                            ca[i * n - 1] = pt[i];
                    }

                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < i; j++)
                            pa[i * n + j] = pa[j * n + i];

                    for (int i = 0; i < n; i++)
                    {
                        pt[i] = 0;
                        for (int j = 0; j < n; j++)
                        {
                            if (j < i)
                                pt[i] += pa[j * n + i] * pv[j];
                            else
                                pt[i] += pa[i * n + j] * pv[j];
                        }
                    }
                }
            }
            return t;
        }
        public static double[] InvertSsgjUnsafeL(double[,] a, double[] V)
        {
            int n = a.GetLength(0);
            if (a.GetLength(0) != a.GetLength(1)) return null;
            double[] t = new double[n];
            unsafe
            {
                fixed (double* pa = a)
                fixed (double* pt = t)
                fixed (double* pv = V)
                {
                    double* ca;
                    double g, w;
                    for (int k = 0; k < n; k++)
                    {
                        w = pa[0];
                        if (w == 0.0)
                            return null;
                        int m = n - k - 1;
                        ca = pa;
                        for (int i = 1; i < n; i++)
                        {
                            g = pa[i * n];
                            pt[i] = g / w;
                            if (i <= m)
                                pt[i] = -pt[i];
                            for (int j = 1; j <= i; j++)
                                ca[j - 1] = ca[n + j] + g * pt[j];
                            ca += n;
                        }

                        ca[n - 1] = 1.0 / w;
                        for (int i = 1; i < n; i++)
                            ca[i - 1] = pt[i];
                    }
                    for (int i = 0; i < n; i++)
                    {
                        pt[i] = 0;
                        for (int j = 0; j < n; j++)
                        {
                            if (j < i)
                                pt[i] += pa[i * n + j] * pv[j];
                            else
                                pt[i] += pa[j * n + i] * pv[j];
                        }
                    }
                }
            }
            return t;
        }

        /// <summary>
        /// 对称正定矩阵求逆
        /// </summary>
        /// <param name="a">求逆矩阵</param>
        /// <returns></returns>
        /// 
        public static double[,] InvertSsgjU(double[,] a)
        {
            double w, g;
            int n = a.GetLength(0);
            if (a.GetLength(0) != a.GetLength(1)) return null;
            double[] pTmp = new double[n];
            double[,] Mat = (double[,])a.Clone();
            for (int k = 0; k < n; k++)
            {
                w = Mat[0, 0];
                if (w == 0.0) return null;
                int m = n - k - 1;
                for (int i = 1; i < n; i++)
                {
                    g = Mat[0, i];
                    pTmp[i] = g / w;
                    if (i <= m)
                        pTmp[i] = -pTmp[i];
                    for (int j = 1; j <= i; j++)
                        Mat[j - 1, i - 1] = Mat[j, i] + g * pTmp[j];
                }

                Mat[n - 1, n - 1] = 1.0 / w;
                for (int i = 1; i < n; i++)
                    Mat[i - 1, (n - 1)] = pTmp[i];
            }

            for (int i = 0; i < n - 1; i++)
                for (int j = i + 1; j < n; j++)
                    Mat[j, i] = Mat[i, j];

            return Mat;
        }
        /// <summary>
        /// 对称正定矩阵求逆
        /// </summary>
        /// <param name="a">求逆矩阵</param>
        /// <returns></returns>
        /// 
        public static double[,] InvertSsgjL(double[,] a)
        {
            double w, g;
            int n = a.GetLength(0);
            if (a.GetLength(0) != a.GetLength(1)) return null;
            double[] pTmp = new double[n];
            double[,] Mat = (double[,])a.Clone();
            for (int k = 0; k < n; k++)
            {
                w = Mat[0, 0];
                if (w == 0.0) return null;
                int m = n - k - 1;
                for (int i = 1; i < n; i++)
                {
                    g = Mat[i, 0];
                    pTmp[i] = g / w;
                    if (i <= m)
                        pTmp[i] = -pTmp[i];
                    for (int j = 1; j <= i; j++)
                        Mat[i - 1, j - 1] = Mat[i, j] + g * pTmp[j];
                }

                Mat[n - 1, n - 1] = 1.0 / w;
                for (int i = 1; i < n; i++)
                    Mat[(n - 1), i - 1] = pTmp[i];
            }
            for (int i = 0; i < n - 1; i++)
                for (int j = i + 1; j < n; j++)
                    Mat[i, j] = Mat[j, i];

            return Mat;
        }
        /*
        public static double[,] InvertSsgjL(double[,] a)
        {
            double w, g;
            int n = a.GetLength(0);
            if (a.GetLength(0) != a.GetLength(1)) return null;
            double[] pTmp = new double[n];
            double[,] Mat = (double[,])a.Clone();
            for (int k = 0; k < n; k++)
            {
                w = Mat[0, 0];
                if (w == 0.0) return null;
                int m = n - k - 1;
                for (int i = 1; i < n; i++)
                {
                    g = Mat[i, 0];
                    pTmp[i] = g / w;
                    if (i <= m)
                        pTmp[i] = -pTmp[i];
                    for (int j = 1; j <= i; j++)
                        Mat[(i - 1), j - 1] = Mat[i, j] + g * pTmp[j];
                }

                Mat[n - 1, n - 1] = 1.0 / w;
                for (int i = 1; i < n; i++)
                    Mat[(n - 1), i - 1] = pTmp[i];
            }

            for (int i = 0; i < n - 1; i++)
                for (int j = i + 1; j < n; j++)
                    Mat[i, j] = Mat[j, i];

            return Mat;
        } */

        /// <summary>
        /// 矩阵LU分解
        /// </summary>
        /// <param name="value">分解矩阵</param>
        /// <returns></returns>
        public static double[,] LuDecomposition(double[,] value)
        {
            double[,] lu = (double[,])value.Clone();

            int rows = lu.GetLength(0);
            int cols = lu.GetLength(1);

            /*            int pivotSign = 1;

                        int[] pivotVector = new int[rows];
                        for (int i = 0; i < rows; i++)
                            pivotVector[i] = i;
            */
            var LUcolj = new double[rows];


            unsafe
            {
                fixed (double* LU = lu)
                {
                    // Outer loop.
                    for (int j = 0; j < cols; j++)
                    {
                        // Make a copy of the j-th column to localize references.
                        for (int i = 0; i < rows; i++)
                            LUcolj[i] = lu[i, j];

                        // Apply previous transformations.
                        for (int i = 0; i < rows; i++)
                        {
                            double s = 0;

                            // Most of the time is spent in
                            // the following dot product:
                            int kmax = Math.Min(i, j);
                            double* LUrowi = &LU[i * cols];
                            for (int k = 0; k < kmax; k++)
                                s += LUrowi[k] * LUcolj[k];

                            LUrowi[j] = LUcolj[i] -= s;
                        }

                        // Find pivot and exchange if necessary.
                        /*
                        int p = j;
                        for (int i = j + 1; i < rows; i++)
                        {
                            if (Math.Abs(LUcolj[i]) > Math.Abs(LUcolj[p]))
                                p = i;
                        }
                        if (p != j)
                        {
                            for (int k = 0; k < cols; k++)
                            {
                                var t = lu[p, k];
                                lu[p, k] = lu[j, k];
                                lu[j, k] = t;
                            }

                            int v = pivotVector[p];
                            pivotVector[p] = pivotVector[j];
                            pivotVector[j] = v;

                            pivotSign = -pivotSign;
                        }*/

                        // Compute multipliers.
                        if (j < rows && lu[j, j] != 0)
                        {
                            for (int i = j + 1; i < rows; i++)
                                lu[i, j] /= lu[j, j];
                        }
                    }
                }
            }
            return lu;
        }

        /// <summary>
        /// 矩阵LU分解求逆,须先分解出LU矩阵
        /// </summary>
        /// <param name="lu">求逆矩阵</param>
        /// <returns></returns>
        public static double[,] LuInverseUnsafe(this double[,] lu)
        {
            int rows = lu.GetLength(0);

            // Copy right hand side with pivoting
            var result = new double[rows, rows];
            // Copy right hand side with pivoting
            /*
            for (int i = 0; i < rows; i++)
            {
                int k = pivotVector[i];
                result[i, k] = 1;
            }*/

            unsafe
            {
                fixed (double* LU = lu)
                fixed (double* px = result)
                {
                    double* cpx, cplu;
                    for (int i = 0; i < rows; i++)
                        px[i * rows + i] = 1;
                    // Solve L*Y = B(piv,:)
                    for (int k = 0; k < rows; k++)
                    {
                        for (int i = k + 1; i < rows; i++)
                        {
                            cpx = &px[i * rows];
                            cplu = &LU[i * rows + k];
                            for (int j = 0; j < rows; j++)
                                cpx[j] -= px[k * rows + j] * *cplu;
                        }
                    }

                    // Solve U*X = I;
                    for (int k = rows - 1; k >= 0; k--)
                    {
                        cpx = &px[k * rows];
                        cplu = &LU[k * rows + k];
                        for (int j = 0; j < rows; j++)
                            cpx[j] /= *cplu;

                        for (int i = 0; i < k; i++)
                        {
                            cpx = &px[i * rows];
                            cplu = &LU[i * rows + k];
                            for (int j = 0; j < rows; j++)
                                cpx[j] -= px[k * rows + j] * *cplu;
                        }
                    }
                }
            }
            return result;
        }

        /// <summary>
        /// 矩阵LU分解求逆
        /// </summary>
        /// <param name="lu">求逆矩阵</param>
        /// <returns></returns>
        public static double[,] LuInverse(this double[,] lu)
        {
            int rows = lu.GetLength(0);

            // Copy right hand side with pivoting
            var X = new double[rows, rows];
            for (int i = 0; i < rows; i++)
            {
                //                int k = pivotVector[i];
                X[i, i] = 1;
            }

            // Solve L*Y = B(piv,:)
            for (int k = 0; k < rows; k++)
                for (int i = k + 1; i < rows; i++)
                    for (int j = 0; j < rows; j++)
                        X[i, j] -= X[k, j] * lu[i, k];

            // Solve U*X = I;
            for (int k = rows - 1; k >= 0; k--)
            {
                for (int j = 0; j < rows; j++)
                    X[k, j] /= lu[k, k];

                for (int i = 0; i < k; i++)
                    for (int j = 0; j < rows; j++)
                        X[i, j] -= X[k, j] * lu[i, k];
            }

            return X;
        }
        /// <summary>
        /// Cholesky分解
        /// </summary>
        /// <param name="A">分解矩阵</param>
        /// <returns></returns>
        public static double[,] Cholesky(double[,] A)
        {
            int n = A.GetLength(0);
            double[,] L = (double[,])A.Clone();
            bool positiveDefinite = true;

            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int k = 0; k < j; k++)
                {
                    double t = L[k, j];
                    for (int i = 0; i < k; i++)
                        t -= L[j, i] * L[k, i];
                    t = t / L[k, k];

                    L[j, k] = t;
                    s += t * t;
                }

                s = L[j, j] - s;

                // Use a tolerance for positive-definiteness
                positiveDefinite &= (s > (double)1e-14 * Math.Abs(L[j, j]));

                L[j, j] = (double)Math.Sqrt((double)s);
            }
            for (int j = 0; j < n; j++)
                for (int k = j + 1; k < n; k++)
                    L[j, k] = 0;
            return L;
        }

        public unsafe static double[,] CholeskyUnsafe(double[,] A)
        {
            double[,] L = (double[,])A.Clone();
            int n = L.GetLength(0);
            bool positiveDefinite = true;
            unsafe
            {
                fixed (double* pl = L)
                {
                    double* cp, rp;
                    for (int j = 0; j < n; j++)
                    {
                        cp = &pl[j * n];
                        Double s = 0;
                        for (int k = 0; k < j; k++)
                        {
                            rp = &pl[k * n];
                            Double t = rp[j];
                            for (int i = 0; i < k; i++)
                                t -= cp[i] * rp[i];
                            t = t / rp[k];

                            cp[k] = t;
                            s += t * t;
                        }
                        s = cp[j] - s;
                        // Use a tolerance for positive-definiteness
                        positiveDefinite &= (s > (Double)1e-14 * Math.Abs(cp[j]));
                        cp[j] = (Double)Math.Sqrt((double)s);
                    }
                }
            }
            return L;
        }
        public unsafe static double[] CholeskyInverseUnsafe(double[,] A, double[] V)
        {
            double[,] L = (double[,])A.Clone();
            int n = L.GetLength(0);
            bool positiveDefinite = true;
            unsafe
            {
                fixed (double* pl = L)
                fixed (double* pv = V)
                {
                    double* cp, rp;
                    for (int j = 0; j < n; j++)
                    {
                        cp = &pl[j * n];
                        Double s = 0;
                        for (int k = 0; k < j; k++)
                        {
                            rp = &pl[k * n];
                            Double t = rp[j];
                            for (int i = 0; i < k; i++)
                                t -= cp[i] * rp[i];
                            t = t / rp[k];

                            cp[k] = t;
                            s += t * t;
                        }
                        s = cp[j] - s;
                        // Use a tolerance for positive-definiteness
                        positiveDefinite &= (s > (Double)1e-14 * Math.Abs(cp[j]));
                        cp[j] = (Double)Math.Sqrt((double)s);
                    }
                    for (int j = 0; j < n; j++)
                    {
                        cp = &pl[j * n];
                        double v = pv[j];
                        for (int k = 0; k < j; k++)
                        {
                            v -= cp[k] * pv[k];
                        }
                        pv[j] = v / cp[j];
                    }
                    for (int j = n - 1; j >= 0; j--)
                    {
                        cp = &pl[j * n];
                        double v = pv[j];
                        for (int k = j + 1; k < n; k++)
                        {
                            rp = &pl[k * n];
                            v -= rp[j] * pv[k];
                        }
                        pv[j] = v / cp[j];
                    }
                }
            }
            return V;
        }
        /// <summary>
        /// 矩阵Cholesky分解求逆
        /// </summary>
        /// <param name="L">下三角矩阵</param>
        /// <returns></returns>
        public static double[,] CholeskyInverseUnsafe(double[,] L)
        {
            int n = L.GetLength(0);
            double[,] B = new double[n, n];

            unsafe
            {
                fixed (double* pl = L)
                fixed (double* px = B)
                {
                    double* cpx, cpl;
                    for (int i = 0; i < n; i++)
                        px[i * n + i] = 1;

                    // Solve L*Y = B;
                    for (int k = 0; k < n; k++)
                    {
                        cpx = &px[k * n];
                        cpl = &pl[k * n];

                        for (int j = 0; j < n; j++)
                        {
                            for (int i = 0; i < k; i++)
                                cpx[j] -= px[i * n + j] * cpl[i];
                            cpx[j] /= cpl[k];
                        }
                    }

                    // Solve L'*X = Y;
                    for (int k = n - 1; k >= 0; k--)
                    {
                        cpx = &px[k * n];
                        cpl = &pl[k * n + k];

                        for (int j = 0; j < n; j++)
                        {
                            for (int i = k + 1; i < n; i++)
                                cpx[j] -= px[i * n + j] * pl[i * n + k];

                            cpx[j] /= *cpl;
                        }
                    }
                }
            }

            return B;
        }
        /// <summary>
        /// 矩阵Cholesky分解求逆
        /// </summary>
        /// <param name="L">下三角矩阵</param>
        /// <returns></returns>
        public static double[,] CholeskyInverse(double[,] L)
        {
            int n = L.GetLength(0);
            double[,] B = new double[n, n];
            for (int i = 0; i < n; i++)
                B[i, i] = 1;

            // Solve L*Y = B;
            for (int k = 0; k < n; k++)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int i = 0; i < k; i++)
                        B[k, j] -= B[i, j] * L[k, i];
                    B[k, j] /= L[k, k];
                }
            }

            // Solve L'*X = Y;
            for (int k = n - 1; k >= 0; k--)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int i = k + 1; i < n; i++)
                        B[k, j] -= B[i, j] * L[i, k];

                    B[k, j] /= L[k, k];
                }
            }

            return B;
        }
        /// <summary>
        /// 矩阵LU分解
        /// </summary>
        /// <param name="B">分解矩阵</param>
        /// <returns></returns>
        public static double[,] LUUnsafe(double[,] B)
        {
            double[,] A = (double[,])B.Clone();
            int n = A.GetLength(0);
            unsafe
            {
                fixed (double* pa = A)
                {
                    for (int k = 0; k < n - 1; k++)
                    {
                        for (int i = k + 1; i < n; i++)
                        {
                            A[i, k] /= A[k, k];
                        }
                        for (int i = k + 1; i < n; i++)
                        {
                            double* cpa = &pa[i * n];
                            for (int j = k + 1; j < n; j++)
                            {
                                cpa[j] -= cpa[k] * pa[k * n + j];
                            }
                        }
                    }
                }
            }
            return A;
        }
        /// <summary>
        /// 矩阵LU分解
        /// </summary>
        /// <param name="B">分解矩阵</param>
        /// <returns></returns>
        public static double[,] LU(double[,] B)
        {
            double[,] A = (double[,])B.Clone();
            int n = A.GetLength(0);
            //            double[,] L = new double[n, n];
            //            double[,] U = new double[n, n];
            for (int k = 0; k < n - 1; k++)
            {
                for (int i = k + 1; i < n; i++)
                {
                    A[i, k] /= A[k, k];
                }
                for (int i = k + 1; i < n; i++)
                {
                    for (int j = k + 1; j < n; j++)
                    {
                        A[i, j] -= A[i, k] * A[k, j];
                    }
                }
            }
            /*
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    L[i, j] = A[i, j];
                    U[i, j] = 0;
                }
                L[i, i] = 1;
                U[i, i] = A[i, i];
                for (int j = i + 1; j < n; j++)
                {
                    L[i, j] = 0;
                    U[i, j] = A[i, j];
                }
            }*/
            return A;
        }

        //public static double[] Householder2B(this double[] A,int order, double[] y)
        //{
        //    double mag, alpha;
        //    int m = A.GetLength(0);
        //    if(m!=y.GetLength(0))
        //        return null;
        //    int n = order+1;
        //    double[,] nA = new double[m,n];
        //    for (int i = 0; i < m; i++)
        //    {
        //        nA[i, 0] = 1.0;
        //    }
        //    for (int j = 0; j < m; j++)
        //    {
        //        for (int i = 0; i < order; i++)
        //        {
        //            nA[j, i + 1] = nA[j, i] * A[j];
        //        }
        //    }
        //    double[] u = new double[m], v = new double[m];
        //    double[,] P = new double[m, m], I = new double[m, m];
        //    for (int i = 0; i < m; i++)
        //        I[i, i] = 1.0;
        //    double[,] Q = (double[,])I.Clone();
        //    double[,] R = (double[,])nA.Clone();

        //    for (int i = 0; i < n; i++)
        //    {
        //        u = new double[m];
        //        v = new double[m];

        //        mag = 0.0;
        //        for (int j = i; j < m; j++)
        //        {
        //            u[j] = R[j, i];
        //            mag += u[j] * u[j];
        //        }
        //        mag = Math.Sqrt(mag);

        //        alpha = u[i] < 0 ? mag : -mag;

        //        mag = 0.0;
        //        for (int j = i; j < m; j++)
        //        {
        //            v[j] = j == i ? u[j] + alpha : u[j];
        //            mag += v[j] * v[j];
        //        }
        //        mag = Math.Sqrt(mag);

        //        if (mag < 0.0000000001) continue;

        //        for (int j = i; j < m; j++) v[j] /= mag;

        //        P = I.Subtract((v.Cov()).Multiply(2.0));

        //        R = P.Dot(R);
        //        Q = Q.Dot(P);
        //    }
        //    return InvUpperTRI(R).Dot(Q.Transpose()).Dot(y);
        //}

        public static double[] Householder2B(this double[,] A, double[] y)
        {
            double mag, alpha;
            int m = A.GetLength(0);
            int n = A.GetLength(1);
            double[] u = new double[m], v = new double[m];
            double[,] P = new double[m, m], I = new double[m, m];
            for (int i = 0; i < m; i++)
                I[i, i] = 1.0;
            double[,] Q = (double[,])I.Clone();
            double[,] R = (double[,])A.Clone();

            for (int i = 0; i < n; i++)
            {
                u = new double[m];
                v = new double[m];

                mag = 0.0;
                for (int j = i; j < m; j++)
                {
                    u[j] = R[j, i];
                    mag += u[j] * u[j];
                }
                mag = Math.Sqrt(mag);

                alpha = u[i] < 0 ? mag : -mag;

                mag = 0.0;
                for (int j = i; j < m; j++)
                {
                    v[j] = j == i ? u[j] + alpha : u[j];
                    mag += v[j] * v[j];
                }
                mag = Math.Sqrt(mag);

                if (mag < 0.0000000001) continue;

                for (int j = i; j < m; j++) v[j] /= mag;

                P = I.Subtract((v.Cov()).Multiply(2.0));

                R = P.Dot(R);
                Q = Q.Dot(P);
            }
            return InvUpperTRI(R).Dot(Q.Transpose()).Dot(y);
        }
        public static void HouseholderDecomposition(this double[,] A, out double[,] Q, out double[,] R)
        {
            double mag, alpha;
            int m = A.GetLength(0);
            int n = A.GetLength(1);
            double[] u = new double[m], v = new double[m];
            double[,] P = new double[m, m], I = new double[m, m];
            for (int i = 0; i < m; i++)
                I[i, i] = 1.0;
            Q = (double[,])I.Clone();
            R = (double[,])A.Clone();

            for (int i = 0; i < n; i++)
            {
                u = new double[m];
                v = new double[m];

                mag = 0.0;
                for (int j = i; j < m; j++)
                {
                    u[j] = R[j, i];
                    mag += u[j] * u[j];
                }
                mag = Math.Sqrt(mag);

                alpha = u[i] < 0 ? mag : -mag;

                mag = 0.0;
                for (int j = i; j < m; j++)
                {
                    v[j] = j == i ? u[j] + alpha : u[j];
                    mag += v[j] * v[j];
                }
                mag = Math.Sqrt(mag);

                if (mag < 0.0000000001) continue;

                for (int j = i; j < m; j++) v[j] /= mag;

                P = I.Subtract((v.Cov()).Multiply(2.0));

                R = P.Dot(R);
                Q = Q.Dot(P);
            }
        }
        public static void householderBidiagonalization(this double[,] A, out double[,] Q, out double[,] R, out double[,] S)
        {
            double mag, alpha;
            int m = A.GetLength(0);
            int n = A.GetLength(1);
            double[] u = new double[m], v = new double[m], u_ = new double[n], v_ = new double[n];

            double[,] P = new double[m, m], I = new double[m, m], P_ = new double[n, n], I_ = new double[n, n];

            for (int i = 0; i < m; i++)
                I[i, i] = 1.0;
            for (int i = 0; i < n; i++)
                I_[i, i] = 1.0;

            Q = (double[,])I.Clone();
            R = (double[,])A.Clone();
            S = (double[,])I_.Clone(); ;

            for (int i = 0; i < n; i++)
            {
                u = new double[m]; v = new double[m];

                mag = 0.0;
                for (int j = i; j < m; j++)
                {
                    u[j] = R[j, i];
                    mag += u[j] * u[j];
                }
                mag = Math.Sqrt(mag);

                alpha = u[i] < 0 ? mag : -mag;

                mag = 0.0;
                for (int j = i; j < m; j++)
                {
                    v[j] = j == i ? u[j] + alpha : u[j];
                    mag += v[j] * v[j];
                }
                mag = Math.Sqrt(mag);

                if (mag > 0.0000000001)
                {
                    for (int j = i; j < m; j++) v[j] /= mag;

                    P = I.Subtract((v.Cov()).Multiply(2.0));

                    R = P.Dot(R);
                    Q = Q.Dot(P);
                }
                /////////////////////////
                u_ = new double[n]; v_ = new double[n];

                mag = 0.0;
                for (int j = i + 1; j < n; j++)
                {
                    u_[j] = R[i, j];
                    mag += u_[j] * u_[j];
                }

                mag = Math.Sqrt(mag);

                alpha = 0;
                if (i < n - 1)
                    alpha = u_[i + 1] < 0 ? mag : -mag;

                mag = 0.0;
                for (int j = i + 1; j < n; j++)
                {
                    v_[j] = j == i + 1 ? u_[j] + alpha : u_[j];
                    mag += v_[j] * v_[j];
                }
                mag = Math.Sqrt(mag);

                if (mag > 0.0000000001)
                {
                    for (int j = i + 1; j < n; j++) v_[j] /= mag;

                    P_ = I_.Subtract((v_.Cov()).Multiply(2.0));

                    R = R.Dot(P_);
                    S = P_.Dot(S);
                }
            }
        }


        // Method to form upper
        // triangular matrix
        public static double[,] InvUpperTRI(this double[,] A)
        {
            int i, j, k;
            int n = A.GetLength(1);
            double[,] R = new double[n, n];
            for (j = 0; j < n; j++)
            {
                for (i = 0; i <= (j - 1); i++)
                {
                    for (k = 0; k <= (j - 1); k++)
                    {
                        R[i, j] += R[i, k] * A[k, j];
                    }
                }
                for (k = 0; k <= (j - 1); k++)
                {
                    R[k, j] /= -A[j, j];
                }
                R[j, j] = 1 / A[j, j];
            }
            return R;
        }

        public static double[,] Transpose(this double[] mat)
        {
            int length = mat.Length;
            double[,] A = new double[length, 1];
            for (int i = 0; i < length; i++)
                A[i, 0] = mat[i];
            return A;
        }
        public static double[,] Transpose(this double[,] mat)
        {
            int Height = mat.GetLength(0);
            int Width = mat.GetLength(1);
            double[,] A = new double[Width, Height];
            for (int i = 0; i < Height; i++)
                for (int j = 0; j < Width; j++)
                    A[j, i] = mat[i, j];
            return A;
        }
        /// <summary>
        /// 协方差矩阵
        /// </summary>
        /// <param name="mat">向量</param>
        /// <returns></returns>
        public static double[,] Cov(this double[] mat)
        {
            int length = mat.Length;
            double[,] A = new double[length, length];
            for (int i = 0; i < length; i++)
            {
                for (int j = i; j < length; j++)
                {
                    A[i, j] = mat[i] * mat[j];
                    if (j > i)
                        A[j, i] = mat[i] * mat[j];
                }
            }
            return A;
        }
        public static double[,] Add(this double[,] A, double[,] B, bool InPlace = false)
        {
            int Height = A.GetLength(0);
            int Width = A.GetLength(1);
            if (A.GetLength(0) != B.GetLength(0) || A.GetLength(1) != B.GetLength(1))
                return null;
            for (int i = 0; i < Height; i++)
            {
                for (int j = 0; j < Width; j++)
                    A[i, j] += B[i, j];
            }
            return A;
        }
        public static double[] Add(this double[] A, double[] B, bool InPlace = false)
        {
            int Height = A.Length;
            if (A == null || B == null || A.Length != B.Length)
                return null;
            for (int i = 0; i < Height; i++)
            {
                A[i] += B[i];
            }
            return A;
        }
        public static double[,] Subtract(this double[,] A, double[,] B, bool InPlace = false)
        {
            int Height = A.GetLength(0);
            int Width = A.GetLength(1);
            if (A.GetLength(0) != B.GetLength(0) || A.GetLength(1) != B.GetLength(1))
                return null;
            double[,] mat = new double[Height, Width];
            for (int i = 0; i < Height; i++)
            {
                for (int j = 0; j < Width; j++)
                    mat[i, j] = A[i, j] - B[i, j];
            }
            return mat;
        }
        public static double[] Subtract(this double[] A, double[] B, bool InPlace = false)
        {
            int Height = A.Length;
            if (A.Length != B.Length)
                return null;
            for (int i = 0; i < Height; i++)
            {
                A[i] -= B[i];
            }
            return A;
        }
        public unsafe static double SubSq(this double[] A, double[] B)
        {
            double value = 0;

            fixed (double* ap = A)
            {
                fixed (double* bp = B)
                {
                    for (int i = 0; i < A.Length; i++)
                    {
                        double v = *(ap + i) - *(bp + i);
                        value += v * v;
                    }
                }
            }
            return value;
        }
        public static double Dot(this double[] A, double[] B)
        {
            if (A == null || B == null)
                return 0;
            int Height = A.Length;
            if (A.Length != B.Length)
                return 0;
            double value = 0;
            for (int i = 0; i < Height; i++)
            {
                value += A[i] * B[i];
            }
            return value;
        }
        public unsafe static double[] DotV(this double[] A, double[] B)
        {
            double[] c = new double[A.Length];
            fixed (double* cp = c)
            {
                fixed (double* ap = A)
                {
                    fixed (double* bp = B)
                    {
                        for (int i = 0; i < A.Length; i++)
                        {
                            *(cp + i) = *(ap + i) * *(bp + i);
                        }
                    }
                }
            }
            return c;
        }
        public static double[] Dot(this double[,] A, double[] B, bool InPlace = false)
        {
            if (A == null || B == null)
                return null;
            int Height = A.GetLength(0);
            int Width = A.GetLength(1);
            if (Width != B.Length)
                return null;
            double[] mat = new double[Height];
            for (int i = 0; i < Height; i++)
            {
                for (int j = 0; j < Width; j++)
                    mat[i] += A[i, j] * B[j];
            }
            return mat;
        }
        public static double[] InnerProduct(this double[] A, double[] B, bool InPlace = false)
        {
            if (A == null || B == null)
                return null;
            int Height = A.Length;
            double[] mat = new double[Height];
            for (int i = 0; i < Height; i++)
            {
                mat[i] = A[i] * B[i];
            }
            return mat;
        }
        public static double[] Multiply(this double[] A, double B, bool InPlace = false)
        {
            int Height = A.Length;
            double[] mat = new double[Height];
            for (int i = 0; i < Height; i++)
            {
                mat[i] = A[i] * B;
            }
            return mat;
        }
        public static double[] Multiply(this double[] A, double[] B, bool InPlace = false)
        {
            int Height = A.Length;
            double[] mat = new double[Height];
            for (int i = 0; i < Height; i++)
            {
                mat[i] = A[i] * B[i];
            }
            return mat;
        }
        public static double[,] Multiply(this double[,] A, double B, bool InPlace = false)
        {
            int Height = A.GetLength(0);
            int Width = A.GetLength(1);
            for (int i = 0; i < Height; i++)
            {
                for (int j = 0; j < Width; j++)
                    A[i, j] = A[i, j] * B;
            }
            return A;
        }
        public static double[,] Divide(this double[,] A, double B)
        {
            int Height = A.GetLength(0);
            int Width = A.GetLength(1);
            for (int i = 0; i < Height; i++)
            {
                for (int j = 0; j < Width; j++)
                    A[i, j] = A[i, j] / B;
            }
            return A;
        }
        public static double[] Divide(this double[] A, double B)
        {
            int Height = A.Length;
            for (int i = 0; i < Height; i++)
            {
                A[i] = A[i] / B;
            }
            return A;
        }
        public static double[] Divide(this double[] a, int b, double[] result)
        {
            for (int i = 0; i < a.Length; i++)
                result[i] = (double)((double)a[i] / (double)b);
            return result;
        }

        public static double[] Mean(List<double[]> A)
        {
            if (A == null || A.Count == 0)
                return null;
            int Height = A[0].Length;
            double[] mat = new double[Height];
            for (int j = 0; j < Height; j++)
            {
                for (int i = 0; i < A.Count; i++)
                {
                    mat[j] += A[i][j];
                }
                mat[j] /= A.Count;
            }
            return mat;
        }
        public static double[,] Mean(List<double[,]> A)
        {
            if (A == null || A.Count == 0)
                return null;
            int Height = A[0].GetLength(0);
            int Width = A[0].GetLength(1);
            double[,] mat = new double[Height, Width];
            for (int i = 0; i < Height; i++)
                for (int j = 0; j < Height; j++)
                {
                    for (int k = 0; k < A.Count; k++)
                    {
                        mat[i, j] += A[k][i, j];
                    }
                    mat[i, j] /= A.Count;
                }
            return mat;
        }
        public static double Mean(this double[] A)
        {
            if (A == null || A.Length == 0)
                return 0;
            double value = 0.0; ;

            for (int i = 0; i < A.Length; i++)
            {
                value += A[i];
            }
            value /= A.Length;
            return value;
        }
        public static double[,] Dot(this double[,] a, double[,] b)
        {
            return Dot(a, b, new double[a.GetLength(0), b.GetLength(1)]);
        }
        public static double[,] Dot(this double[,] a, double[,] b, double[,] result)
        {
            int N = result.GetLength(0);
            int K = a.GetLength(1);
            int M = result.GetLength(1);
            int stride = b.GetLength(1);
            var t = new double[K];

            unsafe
            {
                fixed (double* A = a)
                fixed (double* B = b)
                fixed (double* T = t)
                fixed (double* R = result)
                {
                    for (int j = 0; j < M; j++)
                    {
                        double* pb = B + j;
                        for (int k = 0; k < K; k++)
                        {
                            T[k] = *pb;
                            pb += stride;
                        }

                        double* pa = A;
                        double* pr = R + j;
                        for (int i = 0; i < N; i++)
                        {
                            double s = (double)0;
                            for (int k = 0; k < K; k++)
                                s += (double)((double)pa[k] * (double)T[k]);
                            *pr = (double)s;
                            pa += K;
                            pr += M;
                        }
                    }
                }
            }
            return result;
        }
        public static double[,] DotWithTransposed(this double[,] a, double[,] b)
        {
            return DotWithTransposed(a, b, new double[a.GetLength(0), b.GetLength(0)]);
        }
        public static double[,] DotWithTransposed(this double[,] a, double[,] b, double[,] result)
        {
            int n = a.GetLength(1);
            int m = a.GetLength(0);
            int p = b.GetLength(0);

            unsafe
            {
                fixed (double* A = a)
                fixed (double* B = b)
                fixed (double* R = result)
                {
                    double* pr = R;
                    for (int i = 0; i < m; i++)
                    {
                        double* pb = B;
                        for (int j = 0; j < p; j++, pr++)
                        {
                            double* pa = A + n * i;

                            double s = (double)0;
                            for (int k = 0; k < n; k++)
                                s += (double)((double)(*pa++) * (double)(*pb++));
                            *pr = (double)s;
                        }
                    }
                }
            }
            return result;
        }
        public static double[,] DotWithDiagonal(this double[,] a, double[] b)
        {
            return DotWithDiagonal(a, b, new double[a.GetLength(0), b.Length]);
        }
        public static double[,] DotWithDiagonal(this double[,] a, double[] diagonal, double[,] result)
        {
            int rows = a.GetLength(0);

            unsafe
            {
                fixed (double* ptrA = a)
                fixed (double* ptrR = result)
                {
                    double* A = ptrA;
                    double* R = ptrR;
                    for (int i = 0; i < rows; i++)
                        for (int j = 0; j < diagonal.Length; j++)
                            *R++ = (double)((double)(*A++) * (double)diagonal[j]);
                }
            }
            return result;
        }
        public static double Det(this double[,] Mat)
        {
            int S, k, k1, i, j;
            double[,] DArray;
            double save, ArrayK, tmpDet;
            int Rows = Mat.GetUpperBound(0), Cols = Mat.GetUpperBound(0);
            DArray = (double[,])Mat.Clone();

            if (Rows != Cols) { };
            S = Rows;
            tmpDet = 1;

            for (k = 0; k <= S; k++)
            {
                if (DArray[k, k] == 0)
                {
                    j = k;
                    while ((j < S) && (DArray[k, j] == 0)) j = j + 1;
                    if (DArray[k, j] == 0) return 0;
                    else
                    {
                        for (i = k; i <= S; i++)
                        {
                            save = DArray[i, j];
                            DArray[i, j] = DArray[i, k];
                            DArray[i, k] = save;
                        }
                    }
                    tmpDet = -tmpDet;
                }
                ArrayK = DArray[k, k];
                tmpDet = tmpDet * ArrayK;
                if (k < S)
                {
                    k1 = k + 1;
                    for (i = k1; i <= S; i++)
                    {
                        for (j = k1; j <= S; j++)
                            DArray[i, j] = DArray[i, j] - DArray[i, k] * (DArray[k, j] / ArrayK);
                    }
                }
            }
            return tmpDet;
        }

        public static double[] Dot(this double[] rowVector, double[,] b)
        {
            return Dot(rowVector, b, new double[b.GetLength(1)]);
        }
        public static double[] Dot(this double[] rowVector, double[,] matrix, double[] result)
        {
            int cols = matrix.GetLength(1);
            for (int j = 0; j < cols; j++)
            {
                double s = (double)0;
                for (int k = 0; k < rowVector.Length; k++)
                    s += (double)((double)rowVector[k] * (double)matrix[k, j]);
                result[j] = (double)s;
            }
            return result;
        }
        public static double[] Pow(this double[] value, double y)
        {
            return Pow(value, y, new double[value.Length]);
        }
        public static double[] Pow(this double[] value, double y, double[] result)
        {
            for (int i = 0; i < value.Length; i++)
            {
                var v = value[i];
                result[i] = (double)(Math.Pow((double)Math.Abs(v), y));
            }

            return result;
        }

        public static double[,] Kronecker(this double[,] a, double[,] b, double[,] result)
        {
            int arows = a.GetLength(0);
            int acols = a.GetLength(1);

            int brows = b.GetLength(0);
            int bcols = b.GetLength(1);

            //int crows = arows * brows;
            int ccols = acols * bcols;
            int block = brows * ccols;

            unsafe
            {
                fixed (double* ptrR = result)
                fixed (double* ptrA = a)
                fixed (double* ptrB = b)
                {
                    double* A = ptrA;
                    double* Ri = ptrR;

                    for (int i = 0; i < arows; Ri += block, i++)
                    {
                        double* Rj = Ri;

                        for (int j = 0; j < acols; j++, Rj += bcols, A++)
                        {
                            double* R = Rj;
                            double* B = ptrB;

                            for (int k = 0; k < brows; k++, R += ccols)
                            {
                                for (int l = 0; l < bcols; l++, B++)
                                    *(R + l) = (double)((double)(*A) * (double)(*B));
                            }
                        }
                    }
                }
            }

            return result;
        }
        public static double[] Normalize(this double[] a)
        {
            double[] result = (double[])a.Clone();
            double norm = Math.Sqrt(a.Dot(a));
            return result.Divide(norm);
        }
        public static int MaxIndexOf<T>(this T[] input)
        {
            var max = input.Max();
            int index = Array.IndexOf(input, max);
            return index;
        }
        public static Double[,] PseudoInverse(this Double[,] matrix)
        {
            return new SingularValueDecomposition(matrix,
                computeLeftSingularVectors: true,
                computeRightSingularVectors: true,
                autoTranspose: true).Inverse();
        }
        public static double Hypotenuse(double a, double b)
        {
            double r = 0.0;
            double absA = System.Math.Abs(a);
            double absB = System.Math.Abs(b);

            if (absA > absB)
            {
                r = b / a;
                r = absA * System.Math.Sqrt(1 + r * r);
            }
            else if (b != 0)
            {
                r = a / b;
                r = absB * System.Math.Sqrt(1 + r * r);
            }

            return r;
        }



        /// <summary>
        /// Decomposes a matrix using Doolitle LUP decomposition
        /// </summary>
        /// <param name="matrix">The matrix to decompose (must be square).</param>
        /// <param name="pVect">The row permutations.</param>
        /// <param name="toggle">Tracks row swaps and returns even or odd (+1 or -1).</param>
        /// <returns>The decomposed matrix.</returns>
        internal static double[,] DecomposeMatrix(double[,] matrix, out int[] pVect, out int toggle)
        {
            //Matrix must be square
            if (matrix.GetLength(0) != matrix.GetLength(1))
            {
                throw new ArgumentException("Only square matrices may be decomposed!", "matrix");
            }

            //Doolittle LUP composition
            int n = matrix.GetLength(0);
            double[,] result = CopyMatrix(matrix);

            //Initialize the permutation vector
            pVect = new int[n];
            for (int i = 0; i < n; i++)
            {
                pVect[i] = i;
            }

            //Initialize the toggle
            toggle = 1;

            for (int j = 0; j < n - 1; j++) //Loop through the columns
            {
                //Find the largest value in column j
                double colMax = Math.Abs(result[j, j]);
                int pRow = j;
                for (int i = j + 1; i < n; i++)
                {
                    if (result[i, j] > colMax)
                    {
                        colMax = result[i, j];
                        pRow = i;
                    }
                }

                //If the largest value is not on the pivot, swap the rows
                if (pRow != j)
                {
                    double temp;
                    for (int k = 0; k < n; k++)
                    {
                        temp = result[pRow, k];
                        result[pRow, k] = result[j, k];
                        result[j, k] = temp;
                    }

                    //Swap the permutation information
                    int tempRow = pVect[pRow];
                    pVect[pRow] = pVect[j];
                    pVect[j] = tempRow;

                    //Adjust the row-swap toggle
                    toggle = -toggle;
                }

                if (Math.Abs(result[j, j]) < 1.0E-20)
                {
                    throw new ArgumentException("The matrix is not decomposable.", "matrix");
                }

                for (int i = j + 1; i < n; i++)
                {
                    result[i, j] /= result[j, j];
                    for (int k = j + 1; k < n; k++)
                    {
                        result[i, k] -= result[i, j] * result[j, k];
                    }
                }
            }

            return result;
        }

        //Helper method used to assist in finding the inverse
        private static double[] HelperSolve(double[,] luMatrix, double[] b)
        {
            //Solves luMatrix * x = b
            int n = luMatrix.GetLength(0);
            double[] x = new double[n];
            b.CopyTo(x, 0);

            for (int i = 1; i < n; i++)
            {
                double sum = x[i];
                for (int j = 0; j < i; j++)
                {
                    sum -= luMatrix[i, j] * x[j];
                }
                x[i] = sum;
            }

            x[n - 1] /= luMatrix[n - 1, n - 1];
            for (int i = n - 2; i >= 0; i--)
            {
                double sum = x[i];
                for (int j = i + 1; j < n; j++)
                {
                    sum -= luMatrix[i, j] * x[j];
                }
                x[i] = sum / luMatrix[i, i];
            }

            return x;
        }

        internal static double[,] InverseMatrix(double[,] matrix)
        {
            int n = matrix.GetLength(0);
            double[,] result = CopyMatrix(matrix);

            int[] perm;
            int toggle;
            double[,] lum = DecomposeMatrix(matrix, out perm, out toggle);

            double[] b = new double[n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i == perm[j])
                    {
                        b[j] = 1.0F;
                    }
                    else
                    {
                        b[j] = 0.0F;
                    }
                }

                double[] x = HelperSolve(lum, b);

                for (int j = 0; j < n; j++)
                {
                    result[j, i] = x[j];
                }
            }

            return result;
        }

        //This saves a decomposition step when both the determinant and inverse are needed for the same matrix
        internal static double[,] InverseMatrixandDeterminante(double[,] matrix, out double determinant)
        {
            int n = matrix.GetLength(0);
            double[,] result = CopyMatrix(matrix);

            int[] perm;
            int toggle;
            double[,] lum = DecomposeMatrix(matrix, out perm, out toggle);

            //Compute the inverse
            double[] b = new double[n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i == perm[j])
                    {
                        b[j] = 1.0F;
                    }
                    else
                    {
                        b[j] = 0.0F;
                    }
                }

                double[] x = HelperSolve(lum, b);

                for (int j = 0; j < n; j++)
                {
                    result[j, i] = x[j];
                }
            }

            //Compute the determinant
            determinant = toggle;
            for (int i = 0; i < lum.GetLength(0); i++)
            {
                determinant *= lum[i, i];
            }

            return result;
        }

        internal static double Determinant(double[,] matrix)
        {
            int[] perm;
            int toggle;
            double[,] lum = DecomposeMatrix(matrix, out perm, out toggle);

            double result = toggle;
            for (int i = 0; i < lum.GetLength(0); i++)
            {
                result *= lum[i, i];
            }

            return result;
        }

        internal static double[] ScalarTimesVector(double A, double[] B)
        {
            double[] result = new double[B.Length];

            for (int i = 0; i < B.Length; i++)
            {
                result[i] = A * B[i];
            }

            return result;
        }
        internal static double[] VectorTimesMatrix(double[,] matrix, double[] vector)
        {
            if (vector.Length != matrix.GetLength(1))
            {
                throw new ArgumentException("The vector length must equal the number of rows in the matrix.");
            }

            double[] result = new double[matrix.GetLength(1)];

            for (int i = 0; i < result.Length; i++)
            {
                double tempResult = 0.0;
                for (int j = 0; j < vector.Length; j++)
                {
                    tempResult += vector[j] * matrix[i, j];
                }
                result[i] = tempResult;
            }

            return result;
        }

        //Note: identity matrices must be square
        internal static double[,] IdentMatrix(int size)
        {
            double[,] result = new double[size, size];
            for (int i = 0; i < size; i++)
            {
                result[i, i] = 1.0f;
            }
            return result;
        }

        internal static double[,] CopyMatrix(double[,] matrix)
        {
            double[,] result = new double[matrix.GetLength(0), matrix.GetLength(1)];
            Array.Copy(matrix, result, matrix.Length);
            return result;
        }
    }
}