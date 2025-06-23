using System;
using System.Collections.Generic;

namespace Utilities
{
    public static class GenericArrayExtensions
    {
        public static T[,] Diagonal<T>(int rows, int cols, T[] values)
        {
            return Diagonal(rows, cols, values, new T[rows, cols]);
        }
        public static T[,] Diagonal<T>(int rows, int cols, T[] values, T[,] result)
        {
            int size = Math.Min(rows, Math.Min(cols, values.Length));
            for (int i = 0; i < size; i++)
                result[i, i] = values[i];
            return result;
        }
        public static T[,] Transpose<T>(this T[,] matrix, bool inPlace)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (inPlace)
            {
                if (rows != cols)
                    throw new ArgumentException("Only square matrices can be transposed in place.", "matrix");

                for (int i = 0; i < rows; i++)
                {
                    for (int j = i; j < cols; j++)
                    {
                        T element = matrix[j, i];
                        matrix[j, i] = matrix[i, j];
                        matrix[i, j] = element;
                    }
                }
                return matrix;
            }
            else
            {
                T[,] result = new T[cols, rows];
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        result[j, i] = matrix[i, j];

                return result;
            }
        }
        public static T[,] Copy<T>(this T[,] a)
        {
            return (T[,])a.Clone();
        }
        public static double WeightedAverage<T>(this IEnumerable<T> collection, Func<T, int, double> valueSelector, Func<T, int, double> weightSelector)
        {
            return collection.WeightedAverage(valueSelector, weightSelector, () => 0);
        }
        public static double WeightedAverage<T>(this IEnumerable<T> collection, Func<T, int, double> valueSelector, Func<T, int, double> weightSelector, Func<double> divisionByZeroResolver)
        {
            double weightedSum = 0, sumOfWeights = 0;

            int idx = 0;
            foreach (var item in collection)
            {
                var val = valueSelector(item, idx);
                var w = weightSelector(item, idx);

                weightedSum += val * w;
                sumOfWeights += w;

                idx++;
            }

            if (weightedSum == 0)
                return divisionByZeroResolver();
            else
                return weightedSum / sumOfWeights;
        }
    }
}