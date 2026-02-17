using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Drawing;

namespace Utilities
{
    public class NativeInterop
    {
        public static IntPtr constant = Marshal.StringToHGlobalAuto("3FHJ7j81fAmoEKMirFyKOzNaEIvgQ2lyR1tMRisdHY0uZlgVk49VdJo0ln6bEyWE");
        //byte[] s = Encoding.Default.GetBytes("3FHJ7j81fAmoEKMirFyKOzNaEIvgQ2lyR1tMRisdHY0uZlgVk49VdJo0ln6bEyWE");
        [DllImport("Native")]
        public static extern IntPtr RunComputeOSC(IntPtr ES, IntPtr A, IntPtr B, IntPtr C, IntPtr D, IntPtr E, int BType,int LIndex,int LastIndex);
        [DllImport("Native")]
        public static extern double RunDwell(IntPtr ES, IntPtr A, int LIndex, int LastIndex);

        [DllImport("Native")]
        public static extern double RunNewtonOZ(IntPtr ES, IntPtr A, IntPtr n);
        [DllImport("Native")]
        public static extern double RunNewtonIZ(IntPtr ES, IntPtr A, IntPtr n);
        [DllImport("Native")]
        public static extern int RunPCount(IntPtr ES, double A, double B);
        [DllImport("Native")]
        public static extern IntPtr RunCardinalSpline(IntPtr ES, IntPtr A, int Length, int Samples);

        [DllImport("Native")]
        public static extern void FreeMatrixD(IntPtr A);

    }
    public class CaInvoke
    {


        public static double[] ComputeOSC(double[] A, double[] B, double[] C, double[] D, double[] E, int BType,int LIndex,int LastIndex=0)
        {
            IntPtr result;

            if (A == null)
            {
                throw new ArgumentException("The vectors must be of size 3 to compute a cross product.");
            }
            LastIndex= LastIndex==0?0:LastIndex;
            unsafe
            {
                fixed (double* aPtr = A, bPtr = B, cPtr = C, dPtr = D, ePtr = E)
                {
                    IntPtr aIntPtr = new IntPtr(aPtr);
                    IntPtr bIntPtr = new IntPtr(bPtr);
                    IntPtr cIntPtr = new IntPtr(cPtr);
                    IntPtr dIntPtr = new IntPtr(dPtr);
                    IntPtr eIntPtr = new IntPtr(ePtr);
                    result = NativeInterop.RunComputeOSC(NativeInterop.constant, aIntPtr, bIntPtr, cIntPtr, dIntPtr, eIntPtr, BType,(UInt16)LIndex,(UInt16)LastIndex);
                }
            }
            double[] v = new double[6];
            Marshal.Copy(result, v, 0, 6);
            NativeInterop.FreeMatrixD(result);
            return v;
        }

        public static double Dwell(double[] A, int LIndex, int LastIndex = 0)
        {
            double result;
            if (A == null)
            {
                throw new ArgumentException("The vectors must be of size 3 to compute a cross product.");
            }
            unsafe
            {
                fixed (double* aPtr = A)
                {
                    IntPtr aIntPtr = new IntPtr(aPtr);
                    result = NativeInterop.RunDwell(NativeInterop.constant, aIntPtr, LIndex, LastIndex);
                }
            }
            return result;
        }
        public static PointF[] CardinalSpline(PointF[] Points, int Samples)
        {
            IntPtr result;

            if (Points == null)
            {
                throw new ArgumentException("The vectors must be of size 3 to compute a cross product.");
            }
            int len = Points.Length;
            float[] A = new float[len * 2];
            for (int i = 0; i < len; i++)
            {
                A[i * 2] = Points[i].X;
                A[i * 2 + 1] = Points[i].Y;
            }
            unsafe
            {
                fixed (float* aPtr = A)
                {
                    IntPtr aIntPtr = new IntPtr(aPtr);
                    result = NativeInterop.RunCardinalSpline(NativeInterop.constant, aIntPtr, len, Samples);
                }
            }
            int vc = ((len - 1) * (Samples + 1) + 1);
            float[] v = new float[vc * 2];
            PointF[] rev = new PointF[vc];
            Marshal.Copy(result, v, 0, vc * 2);
            NativeInterop.FreeMatrixD(result);
            for (int i = 0; i < vc; i++)
            {
                rev[i] = new PointF();
                rev[i].X = v[i * 2];
                rev[i].Y = v[i * 2 + 1];
            }
            return rev;
        }
    }
}
