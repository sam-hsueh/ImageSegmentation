using System;
using System.Collections.Generic;

namespace Utilities
{
    public static class FloatArrayExtensions
    {
        public static float[] CholeskyDecompositionInverse3(float[,] source, float[] V, float lambda = 0.0003f)
        {
            float A11 = source[0, 0] + lambda;
            float A22 = source[1, 1] + lambda;
            float A33 = source[2, 2] + lambda;
            //         Cholesly decomposition
            float G11 = (float)Math.Sqrt(A11);
            float G12 = source[0, 1] / G11;
            float G13 = source[0, 2] / G11;

            float G22 = (float)Math.Sqrt(A22 - G12 * G12);
            float G23 = (source[1, 2] - G12 * G13) / G22;

            float G33 = (float)Math.Sqrt(A33 - G13 * G13 - G23 * G23);

            float D1 = V[0] / G11;
            float D2 = (V[1] - G12 * D1) / G22;
            float D3 = (V[2] - G13 * D1 - G23 * D2) / G33;

            float R1 = D3 / G33;
            float R2 = (D2 - G23 * R1) / G22;
            float R3 = (D1 - G12 * R2 - G13 * R1) / G11;
            return new float[] { R3, R2, R1 };
        }
        public static float[] CholeskyDecompositionInverse4(float[,] source, float[] V)
        {
            source[0, 0] = (float)Math.Sqrt(source[0, 0]);
            source[1, 0] = source[0, 1] / source[0, 0];
            source[2, 0] = source[0, 2] / source[0, 0];
            source[3, 0] = source[0, 3] / source[0, 0];

            float G22 = (float)Math.Sqrt(source[1, 1] - source[1, 0] * source[1, 0]);
            float G23 = (source[1, 2] - source[1, 0] * source[2, 0]) / G22;
            float G24 = (source[1, 3] - source[1, 0] * source[3, 0]) / G22;

            float G33 = (float)Math.Sqrt(source[2, 2] - source[2, 0] * source[2, 0] - G23 * G23);
            float G34 = (source[2, 3] - source[2, 0] * source[3, 0] - G23 * G24) / G33;

            float G44 = (float)Math.Sqrt(source[3, 3] - source[3, 0] * source[3, 0] - G24 * G24 - G34 * G34);

            float D1 = V[0] / source[0, 0];
            float D2 = (V[1] - source[1, 0] * D1) / G22;
            float D3 = (V[2] - source[2, 0] * D1 - G23 * D2) / G33;
            float D4 = (V[3] - source[3, 0] * D1 - G24 * D2 - G34 * D3) / G44;

            V[3] = D4 / G44;
            V[2] = (D3 - G34 * V[3]) / G33;
            V[1] = (D2 - G23 * V[2] - G24 * V[3]) / G22;
            V[0] = (D1 - source[1, 0] * V[1] - source[2, 0] * V[2] - source[3, 0] * V[3]) / source[0, 0];
            return V;
        }

        public static float[] CholeskyDecompositionInverse40(float[,] source, float[] V)
        {
            float G11 = (float)Math.Sqrt(source[0, 0]);
            float G12 = source[0, 1] / G11;
            float G13 = source[0, 2] / G11;
            float G14 = source[0, 3] / G11;

            float G22 = (float)Math.Sqrt(source[1, 1] - G12 * G12);
            float G23 = (source[1, 2] - G12 * G13) / G22;
            float G24 = (source[1, 3] - G12 * G14) / G22;

            float G33 = (float)Math.Sqrt(source[2, 2] - G13 * G13 - G23 * G23);
            float G34 = (source[2, 3] - G13 * G14 - G23 * G24) / G33;

            float G44 = (float)Math.Sqrt(source[3, 3] - G14 * G14 - G24 * G24 - G34 * G34);

            float D1 = V[0] / G11;
            float D2 = (V[1] - G12 * D1) / G22;
            float D3 = (V[2] - G13 * D1 - G23 * D2) / G33;
            float D4 = (V[3] - G14 * D1 - G24 * D2 - G34 * D3) / G44;

            V[3] = D4 / G44;
            V[2] = (D3 - G34 * V[3]) / G33;
            V[1] = (D2 - G23 * V[2] - G24 * V[3]) / G22;
            V[0] = (D1 - G12 * V[1] - G13 * V[2] - G14 * V[3]) / G11;
            return V;
        }
        public static float[] CholeskyDecompositionInverse4Unsafe(float[,] source, float[] V, float[] b = null)
        {
            unsafe
            {
                fixed (float* a = source)
                fixed (float* v = V)
                {
                    float* ca = a;
                    float G11 = (float)Math.Sqrt(ca[0]);
                    float G12 = ca[1] / G11;
                    float G13 = ca[2] / G11;
                    float G14 = ca[3] / G11;
                    ca += 4;
                    float G22 = (float)Math.Sqrt(ca[1] - G12 * G12);
                    float G23 = (ca[2] - G12 * G13) / G22;
                    float G24 = (ca[3] - G12 * G14) / G22;
                    ca += 4;
                    float G33 = (float)Math.Sqrt(ca[2] - G13 * G13 - G23 * G23);
                    float G34 = (ca[3] - G13 * G14 - G23 * G24) / G33;
                    ca += 4;
                    float G44 = (float)Math.Sqrt(ca[3] - G14 * G14 - G24 * G24 - G34 * G34);


                    v[0] = v[0] / G11;
                    v[1] = (v[1] - G12 * v[0]) / G22;
                    v[2] = (v[2] - G13 * v[0] - G23 * v[1]) / G33;
                    v[3] = (v[3] - G14 * v[0] - G24 * v[1] - G34 * v[2]) / G44;

                    v[3] = v[3] / G44;
                    v[2] = (v[2] - G34 * v[3]) / G33;
                    v[1] = (v[1] - G23 * v[2] - G24 * v[3]) / G22;
                    v[0] = (v[0] - G12 * v[1] - G13 * v[2] - G14 * v[3]) / G11;
                    if (b != null)
                        fixed (float* pb = b)
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
        public static float[] CholeskyDecompositionInverse40(float[,] source, float[] V, float[] b = null)
        {
            source[0, 0] = (float)Math.Sqrt(source[0, 0]);
            source[1, 0] = source[0, 1] / source[0, 0];
            source[2, 0] = source[0, 2] / source[0, 0];
            source[3, 0] = source[0, 3] / source[0, 0];

            source[1, 1] = (float)Math.Sqrt(source[1, 1] - source[1, 0] * source[1, 0]);
            source[2, 1] = (source[1, 2] - source[1, 0] * source[2, 0]) / source[1, 1];
            source[3, 1] = (source[1, 3] - source[1, 0] * source[3, 0]) / source[1, 1];

            source[2, 2] = (float)Math.Sqrt(source[2, 2] - source[2, 0] * source[2, 0] - source[2, 1] * source[2, 1]);
            source[3, 2] = (source[2, 3] - source[2, 0] * source[3, 0] - source[2, 1] * source[3, 1]) / source[2, 2];

            source[3, 3] = (float)Math.Sqrt(source[3, 3] - source[3, 0] * source[3, 0] - source[3, 1] * source[3, 1] - source[3, 2] * source[3, 2]);

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
        public static float[] CholeskyDecompositionInverse4(float[,] source, float[] V, float[] b = null)
        {
            float G11 = (float)Math.Sqrt(source[0, 0]);
            float G12 = source[0, 1] / G11;
            float G13 = source[0, 2] / G11;
            float G14 = source[0, 3] / G11;

            float G22 = (float)Math.Sqrt(source[1, 1] - G12 * G12);
            float G23 = (source[1, 2] - G12 * G13) / G22;
            float G24 = (source[1, 3] - G12 * G14) / G22;

            float G33 = (float)Math.Sqrt(source[2, 2] - G13 * G13 - G23 * G23);
            float G34 = (source[2, 3] - G13 * G14 - G23 * G24) / G33;

            float G44 = (float)Math.Sqrt(source[3, 3] - G14 * G14 - G24 * G24 - G34 * G34);

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


        public static float[] CholeskyDecompositionInverse5Unsafe(float[,] source, float[] V, float[] b = null)
        {
            unsafe
            {
                fixed (float* a = source)
                fixed (float* v = V)
                {
                    float* ca = a;
                    float G11 = (float)Math.Sqrt(ca[0]);
                    float G12 = ca[1] / G11;
                    float G13 = ca[2] / G11;
                    float G14 = ca[3] / G11;
                    float G15 = ca[4] / G11;
                    ca += 5;
                    float G22 = (float)Math.Sqrt(ca[1] - G12 * G12);
                    float G23 = (ca[2] - G12 * G13) / G22;
                    float G24 = (ca[3] - G12 * G14) / G22;
                    float G25 = (ca[4] - G12 * G15) / G22;
                    ca += 5;
                    float G33 = (float)Math.Sqrt(ca[2] - G13 * G13 - G23 * G23);
                    float G34 = (ca[3] - G13 * G14 - G23 * G24) / G33;
                    float G35 = (ca[4] - G13 * G15 - G23 * G25) / G33;
                    ca += 5;
                    float G44 = (float)Math.Sqrt(ca[3] - G14 * G14 - G24 * G24 - G34 * G34);
                    float G45 = (ca[4] - G14 * G15 - G24 * G25 - G34 * G35) / G44;
                    ca += 5;
                    float G55 = (float)Math.Sqrt(ca[4] - G15 * G15 - G25 * G25 - G35 * G35 - G45 * G45);

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
        public static float[] CholeskyDecompositionInverse50(float[,] source, float[] V, float[] b = null)
        {
            source[0, 0] = (float)Math.Sqrt(source[0, 0]);
            source[1, 0] = source[0, 1] / source[0, 0];
            source[2, 0] = source[0, 2] / source[0, 0];
            source[3, 0] = source[0, 3] / source[0, 0];
            source[4, 0] = source[0, 4] / source[0, 0];

            source[1, 1] = (float)Math.Sqrt(source[1, 1] - source[1, 0] * source[1, 0]);
            source[2, 1] = (source[1, 2] - source[1, 0] * source[2, 0]) / source[1, 1];
            source[3, 1] = (source[1, 3] - source[1, 0] * source[3, 0]) / source[1, 1];
            source[4, 1] = (source[1, 4] - source[1, 0] * source[4, 0]) / source[1, 1];

            source[2, 2] = (float)Math.Sqrt(source[2, 2] - source[2, 0] * source[2, 0] - source[2, 1] * source[2, 1]);
            source[3, 2] = (source[2, 3] - source[2, 0] * source[3, 0] - source[2, 1] * source[3, 1]) / source[2, 2];
            source[4, 2] = (source[2, 4] - source[2, 0] * source[4, 0] - source[2, 1] * source[4, 1]) / source[2, 2];

            source[3, 3] = (float)Math.Sqrt(source[3, 3] - source[3, 0] * source[3, 0] - source[3, 1] * source[3, 1] - source[3, 2] * source[3, 2]);
            source[4, 3] = (source[3, 4] - source[3, 0] * source[4, 0] - source[3, 1] * source[4, 1] - source[3, 2] * source[4, 2]) / source[3, 3];

            source[4, 4] = (float)Math.Sqrt(source[4, 4] - source[4, 0] * source[4, 0] - source[4, 1] * source[4, 1] - source[4, 2] * source[4, 2] - source[4, 3] * source[4, 3]);
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

        public static float[] CholeskyDecompositionInverse5(float[,] source, float[] V, float[] b = null)
        {
            float G11 = (float)Math.Sqrt(source[0, 0]);
            float G12 = source[0, 1] / G11;
            float G13 = source[0, 2] / G11;
            float G14 = source[0, 3] / G11;
            float G15 = source[0, 4] / G11;

            float G22 = (float)Math.Sqrt(source[1, 1] - G12 * G12);
            float G23 = (source[1, 2] - G12 * G13) / G22;
            float G24 = (source[1, 3] - G12 * G14) / G22;
            float G25 = (source[1, 4] - G12 * G15) / G22;

            float G33 = (float)Math.Sqrt(source[2, 2] - G13 * G13 - G23 * G23);
            float G34 = (source[2, 3] - G13 * G14 - G23 * G24) / G33;
            float G35 = (source[2, 4] - G13 * G15 - G23 * G25) / G33;

            float G44 = (float)Math.Sqrt(source[3, 3] - G14 * G14 - G24 * G24 - G34 * G34);
            float G45 = (source[3, 4] - G14 * G15 - G24 * G25 - G34 * G35) / G44;

            float G55 = (float)Math.Sqrt(source[4, 4] - G15 * G15 - G25 * G25 - G35 * G35 - G45 * G45);

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

        public static float[] CholeskyDecompositionInverse5(float[,] source, float[] V, float lambda = 0.03f)
        {
            float A11 = source[0, 0] + lambda;
            float A22 = source[1, 1] + lambda;
            float A33 = source[2, 2] + lambda;
            float A44 = source[3, 3] + lambda;
            float A55 = source[4, 4] + lambda;
            //         Cholesly decomposition
            float G11 = (float)Math.Sqrt(A11);
            float G12 = source[0, 1] / G11;
            float G13 = source[0, 2] / G11;
            float G14 = source[0, 3] / G11;
            float G15 = source[0, 4] / G11;

            float G22 = (float)Math.Sqrt(A22 - G12 * G12);
            float G23 = (source[1, 2] - G12 * G13) / G22;
            float G24 = (source[1, 3] - G12 * G14) / G22;
            float G25 = (source[1, 4] - G12 * G15) / G22;

            float G33 = (float)Math.Sqrt(A33 - G13 * G13 - G23 * G23);
            float G34 = (source[2, 3] - G13 * G14 - G23 * G24) / G33;
            float G35 = (source[2, 4] - G13 * G15 - G23 * G25) / G33;

            float G44 = (float)Math.Sqrt(A44 - G14 * G14 - G24 * G24 - G34 * G34);
            float G45 = (source[3, 4] - G14 * G15 - G24 * G25 - G34 * G35) / G44;

            float G55 = (float)Math.Sqrt(A55 - G15 * G15 - G25 * G25 - G35 * G35 - G45 * G45);

            float D1 = V[0] / G11;
            float D2 = (V[1] - G12 * D1) / G22;
            float D3 = (V[2] - G13 * D1 - G23 * D2) / G33;
            float D4 = (V[3] - G14 * D1 - G24 * D2 - G34 * D3) / G44;
            float D5 = (V[4] - G15 * D1 - G25 * D2 - G35 * D3 - G45 * D4) / G55;

            float R1 = D5 / G55;
            float R2 = (D4 - G45 * R1) / G44;
            float R3 = (D3 - G34 * R2 - G35 * R1) / G33;
            float R4 = (D2 - G23 * R3 - G24 * R2 - G25 * R3) / G22;
            float R5 = (D1 - G12 * R3 - G13 * R2 - G14 * R3 - G15 * R4) / G11;

            return new float[] { R1, R2, R3, R3, R4 };
        }

        /// <summary>
        /// 构造正定矩阵
        /// </summary>
        /// <param name="lu">创建正定矩阵</param>
        /// <returns></returns>
        public static float[,] LeastSquareUnsafe(float[] a)
        {
            int rows = a.Length;

            // Copy right hand side with pivoting
            var result = new float[rows, rows];

            unsafe
            {
                fixed (float* A = a)
                fixed (float* px = result)
                {
                    float* X = px;
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
                        float* X = &px[i * rows];
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
        public static float[,] Multiply(this float[,] a, float[,] b)
        {
            float[,] result;
            int N = a.GetLength(0);
            int M = b.GetLength(1);
            int K = a.GetLength(1);
            var t = new float[K];
            if (K != b.GetLength(0)) return null;

            result = new float[N, M];
            unsafe
            {
                fixed (float* A = a)
                fixed (float* B = b)
                fixed (float* T = t)
                fixed (float* R = result)
                {
                    for (int j = 0; j < M; j++)
                    {
                        float* pb = B + j;
                        for (int k = 0; k < K; k++)
                        {
                            T[k] = *pb;
                            pb += M;
                        }

                        float* pa = A;
                        float* pr = R + j;
                        for (int i = 0; i < N; i++)
                        {
                            float s = (float)0;
                            for (int k = 0; k < K; k++)
                                s += (float)((float)pa[k] * (float)T[k]);
                            *pr = (float)s;
                            pa += K;
                            pr += M;
                        }
                    }
                }
            }
            t = null;
            return result;
        }
        /*
        /// <summary>
        /// 矩阵相乘
        /// </summary>
        /// <param name="a">左矩阵</param>
        /// <param name="b">右矩阵</param>
        /// <returns></returns>
        public static float[,] Multiply(this float[,] a, float[,] b)
        {
            float[,] result;
            int N = a.GetLength(0);
            int M = b.GetLength(1);
            int K = a.GetLength(1);
            var t = new float[K];
            if (K != b.GetLength(0)) return null;

            result = new float[N, M];
            for (int i = 0; i < N; i++)
                for (int j = 0; j < M; j++)
                    for (int k = 0; k < K; k++)
                        result[i, j] += ((float)a[i, k] * (float)b[k, j]);

            return result;
        }*/
        /// <summary>
        /// 对称正定矩阵求逆
        /// </summary>
        /// <param name="a">求逆矩阵</param>
        /// <returns></returns>
        public static float[,] InvertSsgjUnsafeU(float[,] Mat)
        {
            float[,] a = (float[,])Mat.Clone();
            int n = a.GetLength(0);
            if (a.GetLength(0) != a.GetLength(1)) return null;

            float[] t = new float[n];
            unsafe
            {
                fixed (float* pa = a)
                fixed (float* pt = t)
                {
                    float* ca;
                    float g, w;
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

                        ca[n * n - 1] = 1.0f / w;
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
        public static float[,] InvertSsgjUnsafeL(float[,] a)
        {
            int n = a.GetLength(0);
            if (a.GetLength(0) != a.GetLength(1)) return null;
            float[] t = new float[n];
            unsafe
            {
                fixed (float* pa = a)
                fixed (float* pt = t)
                {
                    float* ca;
                    float g, w;
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

                        ca[n - 1] = 1.0f / w;
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

        public static float[] InvertSsgjUnsafeU(float[,] a, float[] V)
        {
            int n = a.GetLength(0);
            if (a.GetLength(0) != a.GetLength(1)) return null;

            float[] t = new float[n];
            unsafe
            {
                fixed (float* pa = a)
                fixed (float* pt = t)
                fixed (float* pv = V)
                {
                    float* ca;
                    float g, w;
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

                        ca[n * n - 1] = 1.0f / w;
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
        public static float[] InvertSsgjUnsafeL(float[,] a, float[] V)
        {
            int n = a.GetLength(0);
            if (a.GetLength(0) != a.GetLength(1)) return null;
            float[] t = new float[n];
            unsafe
            {
                fixed (float* pa = a)
                fixed (float* pt = t)
                fixed (float* pv = V)
                {
                    float* ca;
                    float g, w;
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

                        ca[n - 1] = 1.0f / w;
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
        public static float[,] InvertSsgjU(float[,] a)
        {
            float w, g;
            int n = a.GetLength(0);
            if (a.GetLength(0) != a.GetLength(1)) return null;
            float[] pTmp = new float[n];
            float[,] Mat = (float[,])a.Clone();
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

                Mat[n - 1, n - 1] = 1.0f / w;
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
        public static float[,] InvertSsgjL(float[,] a)
        {
            float w, g;
            int n = a.GetLength(0);
            if (a.GetLength(0) != a.GetLength(1)) return null;
            float[] pTmp = new float[n];
            float[,] Mat = (float[,])a.Clone();
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

                Mat[n - 1, n - 1] = 1.0f / w;
                for (int i = 1; i < n; i++)
                    Mat[(n - 1), i - 1] = pTmp[i];
            }
            for (int i = 0; i < n - 1; i++)
                for (int j = i + 1; j < n; j++)
                    Mat[i, j] = Mat[j, i];

            return Mat;
        }
        /*
        public static float[,] InvertSsgjL(float[,] a)
        {
            float w, g;
            int n = a.GetLength(0);
            if (a.GetLength(0) != a.GetLength(1)) return null;
            float[] pTmp = new float[n];
            float[,] Mat = (float[,])a.Clone();
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
        public static float[,] LuDecomposition(float[,] value)
        {
            float[,] lu = (float[,])value.Clone();

            int rows = lu.GetLength(0);
            int cols = lu.GetLength(1);

            /*            int pivotSign = 1;

                        int[] pivotVector = new int[rows];
                        for (int i = 0; i < rows; i++)
                            pivotVector[i] = i;
            */
            var LUcolj = new float[rows];


            unsafe
            {
                fixed (float* LU = lu)
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
                            float s = 0;

                            // Most of the time is spent in
                            // the following dot product:
                            int kmax = (int)Math.Min(i, j);
                            float* LUrowi = &LU[i * cols];
                            for (int k = 0; k < kmax; k++)
                                s += LUrowi[k] * LUcolj[k];

                            LUrowi[j] = LUcolj[i] -= s;
                        }

                        // Find pivot and exchange if necessary.
                        /*
                        int p = j;
                        for (int i = j + 1; i < rows; i++)
                        {
                            if ((float)Math.Abs(LUcolj[i]) > (float)Math.Abs(LUcolj[p]))
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
        /// 矩阵LU分解求逆
        /// </summary>
        /// <param name="lu">求逆矩阵</param>
        /// <returns></returns>
        public static float[,] LuInverseUnsafe(this float[,] lu)
        {
            int rows = lu.GetLength(0);

            // Copy right hand side with pivoting
            var result = new float[rows, rows];
            // Copy right hand side with pivoting
            /*
            for (int i = 0; i < rows; i++)
            {
                int k = pivotVector[i];
                result[i, k] = 1;
            }*/

            unsafe
            {
                fixed (float* LU = lu)
                fixed (float* px = result)
                {
                    float* cpx, cplu;
                    for (int i = 0; i < rows; i++)
                        px[i * rows + i] = 1f;
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
        public static float[,] LuInverse(this float[,] lu)
        {
            int rows = lu.GetLength(0);

            // Copy right hand side with pivoting
            var X = new float[rows, rows];
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
        public static float[,] Cholesky(float[,] A)
        {
            int n = A.GetLength(0);
            float[,] L = (float[,])A.Clone();
            bool positiveDefinite = true;

            for (int j = 0; j < n; j++)
            {
                float s = 0;
                for (int k = 0; k < j; k++)
                {
                    float t = L[k, j];
                    for (int i = 0; i < k; i++)
                        t -= L[j, i] * L[k, i];
                    t = t / L[k, k];

                    L[j, k] = t;
                    s += t * t;
                }

                s = L[j, j] - s;

                // Use a tolerance for positive-definiteness
                positiveDefinite &= (s > (float)1e-14 * (float)Math.Abs(L[j, j]));

                L[j, j] = (float)(float)Math.Sqrt((float)s);
            }
            for (int j = 0; j < n; j++)
                for (int k = j + 1; k < n; k++)
                    L[j, k] = 0;
            return L;
        }

        public unsafe static float[,] CholeskyUnsafe(float[,] A)
        {
            float[,] L = (float[,])A.Clone();
            int n = L.GetLength(0);
            bool positiveDefinite = true;
            unsafe
            {
                fixed (float* pl = L)
                {
                    float* cp, rp;
                    for (int j = 0; j < n; j++)
                    {
                        cp = &pl[j * n];
                        float s = 0;
                        for (int k = 0; k < j; k++)
                        {
                            rp = &pl[k * n];
                            float t = rp[j];
                            for (int i = 0; i < k; i++)
                                t -= cp[i] * rp[i];
                            t = t / rp[k];

                            cp[k] = t;
                            s += t * t;
                        }
                        s = cp[j] - s;
                        // Use a tolerance for positive-definiteness
                        positiveDefinite &= (s > (float)1e-14 * (float)Math.Abs(cp[j]));
                        cp[j] = (float)Math.Sqrt((float)s);
                    }
                }
            }
            return L;
        }
        public unsafe static float[] CholeskyInverseUnsafe(float[,] A, float[] V)
        {
            float[,] L = (float[,])A.Clone();
            int n = L.GetLength(0);
            bool positiveDefinite = true;
            unsafe
            {
                fixed (float* pl = L)
                fixed (float* pv = V)
                {
                    float* cp, rp;
                    for (int j = 0; j < n; j++)
                    {
                        cp = &pl[j * n];
                        float s = 0;
                        for (int k = 0; k < j; k++)
                        {
                            rp = &pl[k * n];
                            float t = rp[j];
                            for (int i = 0; i < k; i++)
                                t -= cp[i] * rp[i];
                            t = t / rp[k];

                            cp[k] = t;
                            s += t * t;
                        }
                        s = cp[j] - s;
                        // Use a tolerance for positive-definiteness
                        positiveDefinite &= (s > (float)1e-14 * (float)Math.Abs(cp[j]));
                        cp[j] = (float)(float)Math.Sqrt((float)s);
                    }
                    for (int j = 0; j < n; j++)
                    {
                        cp = &pl[j * n];
                        float v = pv[j];
                        for (int k = 0; k < j; k++)
                        {
                            v -= cp[k] * pv[k];
                        }
                        pv[j] = v / cp[j];
                    }
                    for (int j = n - 1; j >= 0; j--)
                    {
                        cp = &pl[j * n];
                        float v = pv[j];
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
        public static float[,] CholeskyInverseUnsafe(float[,] L)
        {
            int n = L.GetLength(0);
            float[,] B = new float[n, n];

            unsafe
            {
                fixed (float* pl = L)
                fixed (float* px = B)
                {
                    float* cpx, cpl;
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
        public static float[,] CholeskyInverse(float[,] L)
        {
            int n = L.GetLength(0);
            float[,] B = new float[n, n];
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
        public static float[,] LUUnsafe(float[,] B)
        {
            float[,] A = (float[,])B.Clone();
            int n = A.GetLength(0);
            unsafe
            {
                fixed (float* pa = A)
                {
                    for (int k = 0; k < n - 1; k++)
                    {
                        for (int i = k + 1; i < n; i++)
                        {
                            A[i, k] /= A[k, k];
                        }
                        for (int i = k + 1; i < n; i++)
                        {
                            float* cpa = &pa[i * n];
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
        public static float[,] LU(float[,] B)
        {
            float[,] A = (float[,])B.Clone();
            int n = A.GetLength(0);
            //            float[,] L = new float[n, n];
            //            float[,] U = new float[n, n];
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
        public static float[,] Transpose(this float[] mat)
        {
            int length = mat.Length;
            float[,] A = new float[length, 1];
            for (int i = 0; i < length; i++)
                A[i, 0] = mat[i];
            return A;
        }
        public static float[,] Transpose(this float[,] mat)
        {
            int Height = mat.GetLength(0);
            int Width = mat.GetLength(1);
            float[,] A = new float[Width, Height];
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
        public static float[,] Cov(this float[] mat)
        {
            int length = mat.Length;
            float[,] A = new float[length, length];
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
        public static float[,] Add(this float[,] A, float[,] B, bool InPlace = false)
        {
            int Height = A.GetLength(0);
            int Width = A.GetLength(1);
            if (A.GetLength(0) != B.GetLength(0) || A.GetLength(1) != B.GetLength(1))
                return null;
            if (InPlace)
            {
                for (int i = 0; i < Height; i++)
                {
                    for (int j = 0; j < Width; j++)
                        A[i, j] += B[i, j];
                }
                return A;
            }
            else
            {
                float[,] A2 = new float[Height, Width];
                for (int i = 0; i < Height; i++)
                {
                    for (int j = 0; j < Width; j++)
                        A2[i, j] = A[i, j] + B[i, j];
                }
                return A2;
            }
        }
        public static float[] Add(this float[] A, float[] B, bool InPlace = false)
        {
            int Height = A.Length;
            if (A == null || B == null || A.Length != B.Length)
                return null;
            if (InPlace)
            {
                for (int i = 0; i < Height; i++)
                {
                    A[i] += B[i];
                }
                return A;

            }
            else
            {
                float[] A2 = new float[Height];
                for (int i = 0; i < Height; i++)
                {
                    A2[i] = A[i] + B[i];
                }
                return A2;
            }
        }
        public static float[,] Subtract(this float[,] A, float[,] B, bool InPlace = false)
        {
            int Height = A.GetLength(0);
            int Width = A.GetLength(1);
            if (A.GetLength(0) != B.GetLength(0) || A.GetLength(1) != B.GetLength(1))
                return null;
            if (InPlace)
            {
                for (int i = 0; i < Height; i++)
                {
                    for (int j = 0; j < Width; j++)
                        A[i, j] -= B[i, j];
                }
                return A;
            }
            else
            {
                float[,] mat = new float[Height, Width];
                for (int i = 0; i < Height; i++)
                {
                    for (int j = 0; j < Width; j++)
                        mat[i, j] = A[i, j] - B[i, j];
                }
                return mat;
            }
        }
        public static float[] Subtract(this float[] A, float[] B, bool InPlace = false)
        {
            int Height = A.Length;
            if (A.Length != B.Length)
                return null;
            if (InPlace)
            {
                for (int i = 0; i < Height; i++)
                {
                    A[i] -= B[i];
                }
                return A;
            }
            else
            {
                float[] C = new float[Height];
                for (int i = 0; i < Height; i++)
                {
                    C[i] = A[i] - B[i];
                }
                return C;
            }
        }
        public unsafe static float SubSq(this float[] A, float[] B)
        {
            float value = 0;

            fixed (float* ap = A)
            {
                fixed (float* bp = B)
                {
                    for (int i = 0; i < A.Length; i++)
                    {
                        float v = *(ap + i) - *(bp + i);
                        value += v * v;
                    }
                }
            }
            return value;
        }
        public static float Dot(this float[] A, float[] B)
        {
            if (A == null || B == null)
                return 0;
            int Height = A.Length;
            if (A.Length != B.Length)
                return 0;
            float value = 0;
            for (int i = 0; i < Height; i++)
            {
                value += A[i] * B[i];
            }
            return value;
        }
        public unsafe static float[] DotV(this float[] A, float[] B)
        {
            float[] c = new float[A.Length];
            fixed (float* cp = c)
            {
                fixed (float* ap = A)
                {
                    fixed (float* bp = B)
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
        public static float[] Dot(this float[,] A, float[] B)
        {
            if (A == null || B == null)
                return null;
            int Height = A.GetLength(0);
            int Width = A.GetLength(1);
            if (Width != B.Length)
                return null;

            float[] mat = new float[Height];
            for (int i = 0; i < Height; i++)
            {
                for (int j = 0; j < Width; j++)
                    mat[i] += A[i, j] * B[j];
            }
            return mat;
        }
        public static float[] InnerProduct(this float[] A, float[] B, bool InPlace = false)
        {
            if (A == null || B == null)
                return null;
            int Height = A.Length;
            if (InPlace)
            {
                for (int i = 0; i < Height; i++)
                {
                    A[i] *= B[i];
                }
                return A;
            }
            else
            {
                float[] mat = new float[Height];
                for (int i = 0; i < Height; i++)
                {
                    mat[i] = A[i] * B[i];
                }
                return mat;
            }
        }
        public static float[] Multiply(this float[] A, float B, bool InPlace = false)
        {
            int Height = A.Length;
            if (InPlace)
            {
                for (int i = 0; i < Height; i++)
                {
                    A[i] *= B;
                }
                return A;
            }
            else
            {
                float[] mat = new float[Height];
                for (int i = 0; i < Height; i++)
                {
                    mat[i] = A[i] * B;
                }
                return mat;
            }
        }
        public unsafe static float[] Multiply(this float[] A, float[] B, bool InPlace = false)
        {
            int Height = A.Length;
            if (InPlace)
            {
                for (int i = 0; i < Height; i++)
                {
                    A[i] *= B[i];
                }
                return A;
            }
            else
            {

                float[] mat = new float[Height];
                fixed (float* ptra = A, ptrb = B, ptrm = mat)
                {
                    float* m = ptrm, a = ptra, b = ptrb;
                    for (int i = 0; i < Height; i++, a++, b++, m++)
                    {
                        *m = *a * *b;
                    }
                }
                return mat;
            }
        }
        public static float[,] Multiply(this float[,] A, float B, bool InPlace = false)
        {
            int Height = A.GetLength(0);
            int Width = A.GetLength(1);
            if (InPlace)
            {
                for (int i = 0; i < Height; i++)
                {
                    for (int j = 0; j < Width; j++)
                        A[i, j] = A[i, j] * B;
                }
                return A;
            }
            else
            {
                float[,] C = new float[Height, Width];
                for (int i = 0; i < Height; i++)
                {
                    for (int j = 0; j < Width; j++)
                        C[i, j] = A[i, j] * B;
                }
                return C;
            }
        }
        public static float[,] Divide(this float[,] A, float B)
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
        public static float[] Divide(this float[] A, float B)
        {
            int Height = A.Length;
            for (int i = 0; i < Height; i++)
            {
                A[i] = A[i] / B;
            }
            return A;
        }
        public static float[] Divide(this float[] a, int b, float[] result)
        {
            for (int i = 0; i < a.Length; i++)
                result[i] = (float)((float)a[i] / (float)b);
            return result;
        }

        public static float[] Mean(List<float[]> A)
        {
            if (A == null || A.Count == 0)
                return null;
            int Height = A[0].Length;
            float[] mat = new float[Height];
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
        public static float[,] Mean(List<float[,]> A)
        {
            if (A == null || A.Count == 0)
                return null;
            int Height = A[0].GetLength(0);
            int Width = A[0].GetLength(1);
            float[,] mat = new float[Height, Width];
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
        public static float Mean(this float[] A)
        {
            if (A == null || A.Length == 0)
                return 0;
            float value = 0.0f;

            for (int i = 0; i < A.Length; i++)
            {
                value += A[i];
            }
            value /= A.Length;
            return value;
        }
        public static float[,] Dot(this float[,] a, float[,] b)
        {
            return Dot(a, b, new float[a.GetLength(0), b.GetLength(1)]);
        }
        public static float[,] Dot(this float[,] a, float[,] b, float[,] result)
        {
            int N = result.GetLength(0);
            int K = a.GetLength(1);
            int M = result.GetLength(1);
            int stride = b.GetLength(1);
            var t = new float[K];

            unsafe
            {
                fixed (float* A = a)
                fixed (float* B = b)
                fixed (float* T = t)
                fixed (float* R = result)
                {
                    for (int j = 0; j < M; j++)
                    {
                        float* pb = B + j;
                        for (int k = 0; k < K; k++)
                        {
                            T[k] = *pb;
                            pb += stride;
                        }

                        float* pa = A;
                        float* pr = R + j;
                        for (int i = 0; i < N; i++)
                        {
                            float s = (float)0;
                            for (int k = 0; k < K; k++)
                                s += (float)((float)pa[k] * (float)T[k]);
                            *pr = (float)s;
                            pa += K;
                            pr += M;
                        }
                    }
                }
            }
            return result;
        }
        public static float[,] DotWithTransposed(this float[,] a, float[,] b)
        {
            return DotWithTransposed(a, b, new float[a.GetLength(0), b.GetLength(0)]);
        }
        public static float[,] DotWithTransposed(this float[,] a, float[,] b, float[,] result)
        {
            int n = a.GetLength(1);
            int m = a.GetLength(0);
            int p = b.GetLength(0);

            unsafe
            {
                fixed (float* A = a)
                fixed (float* B = b)
                fixed (float* R = result)
                {
                    float* pr = R;
                    for (int i = 0; i < m; i++)
                    {
                        float* pb = B;
                        for (int j = 0; j < p; j++, pr++)
                        {
                            float* pa = A + n * i;

                            float s = (float)0;
                            for (int k = 0; k < n; k++)
                                s += (float)((float)(*pa++) * (float)(*pb++));
                            *pr = (float)s;
                        }
                    }
                }
            }
            return result;
        }
        public static float[,] DotWithDiagonal(this float[,] a, float[] b)
        {
            return DotWithDiagonal(a, b, new float[a.GetLength(0), b.Length]);
        }
        public static float[,] DotWithDiagonal(this float[,] a, float[] diagonal, float[,] result)
        {
            int rows = a.GetLength(0);

            unsafe
            {
                fixed (float* ptrA = a)
                fixed (float* ptrR = result)
                {
                    float* A = ptrA;
                    float* R = ptrR;
                    for (int i = 0; i < rows; i++)
                        for (int j = 0; j < diagonal.Length; j++)
                            *R++ = (float)((float)(*A++) * (float)diagonal[j]);
                }
            }
            return result;
        }
        public static float Det(this float[,] Mat)
        {
            int S, k, k1, i, j;
            float[,] DArray;
            float save, ArrayK, tmpDet;
            int Rows = Mat.GetUpperBound(0), Cols = Mat.GetUpperBound(0);
            DArray = (float[,])Mat.Clone();

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

        public static float[] Dot(this float[] rowVector, float[,] b)
        {
            return Dot(rowVector, b, new float[b.GetLength(1)]);
        }
        public static float[] Dot(this float[] rowVector, float[,] matrix, float[] result)
        {
            int cols = matrix.GetLength(1);
            for (int j = 0; j < cols; j++)
            {
                float s = (float)0;
                for (int k = 0; k < rowVector.Length; k++)
                    s += (float)((float)rowVector[k] * (float)matrix[k, j]);
                result[j] = (float)s;
            }
            return result;
        }
        public static float[] Pow(this float[] value, float y)
        {
            return Pow(value, y, new float[value.Length]);
        }
        public static float[] Pow(this float[] value, float y, float[] result)
        {
            for (int i = 0; i < value.Length; i++)
            {
                var v = value[i];
                result[i] = (float)((float)Math.Pow((float)(float)Math.Abs(v), y));
            }

            return result;
        }
        public static float[,] Kronecker(this float[,] a, float[,] b, float[,] result)
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
                fixed (float* ptrR = result)
                fixed (float* ptrA = a)
                fixed (float* ptrB = b)
                {
                    float* A = ptrA;
                    float* Ri = ptrR;

                    for (int i = 0; i < arows; Ri += block, i++)
                    {
                        float* Rj = Ri;

                        for (int j = 0; j < acols; j++, Rj += bcols, A++)
                        {
                            float* R = Rj;
                            float* B = ptrB;

                            for (int k = 0; k < brows; k++, R += ccols)
                            {
                                for (int l = 0; l < bcols; l++, B++)
                                    *(R + l) = (float)((float)(*A) * (float)(*B));
                            }
                        }
                    }
                }
            }

            return result;
        }
        public static float[] Normalize(this float[] a)
        {
            float[] result = (float[])a.Clone();
            float norm = (float)Math.Sqrt(a.Dot(a));
            return result.Divide(norm);
        }
        public static float[,] PseudoInverse(this float[,] matrix)
        {
            return new SingularValueDecompositionF(matrix,
                computeLeftSingularVectors: true,
                computeRightSingularVectors: true,
                autoTranspose: true).Inverse();
        }
        public static float Hypotenuse(float a, float b)
        {
            float r = 0.0f;
            float absA = (float)Math.Abs(a);
            float absB = (float)Math.Abs(b);

            if (absA > absB)
            {
                r = b / a;
                r = absA * (float)Math.Sqrt(1 + r * r);
            }
            else if (b != 0)
            {
                r = a / b;
                r = absB * (float)Math.Sqrt(1 + r * r);
            }

            return r;
        }
    }
}