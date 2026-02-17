using System;

namespace Utilities
{
    public static class JDecompositionExtension
    {

        public static void Factorization(double[][] A, out double[][] s, out double[][] u, out double[][] v)
        {

            double[][] A_t, A_t_x_A, u_1, v_1, s_inverse, av_matrix;
            double[] eigenvalues = new double[A.Length];
            v_1 = new double[A.Length][];
            transpose_matr(A, out A_t);

            prod_matrix(A_t, A, out A_t_x_A);
            GetEigenvalsEigenvecs(A_t_x_A, ref eigenvalues, ref v_1, 0);

            transpose_matr(v_1, out v);

            s = new double[A.Length][];

            int e_size = eigenvalues.Length;
            for (int index = 0; index < e_size; index++)
            {
                s[index] = new double[e_size];
                s[index][index] = eigenvalues[index];
            }

            InverseDiagMatrix(s, out s_inverse);

            prod_matrix(A, v, out av_matrix);
            prod_matrix(av_matrix, s_inverse, out u);
        }



        // Inverse Diagonal Matrix that is not necessary non-degenerate
        public static void InverseDiagMatrix(double[][] A, out double[][] inv_A)
        {
            int m_size = A.Length;

            inv_A = new double[m_size][];

            for (int index = 0; index < m_size; index++)
                inv_A[index] = new double[A[index].Length];

            for (int index = 0; index < m_size; index++)
            {
                if (A[index][index] != 0.0)
                {
                    inv_A[index][index] = 1.0 / A[index][index];
                }
                else
                {
                    inv_A[index][index] = 0.0;
                }
            }
        }


        public static void prod_matrix(double[][] A, double[][] B, out double[][] prod)
        {
            prod = new double[A.Length][];
            for (int row = 0; row < A.Length; row++)
            {
                prod[row] = new double[A[row].Length];
            }

            int m_size = A.Length;

            for (int row = 0; row < m_size; row++)
                for (int col = 0; col < m_size; col++)
                {
                    for (int k = 0; k < m_size; k++)
                        prod[row][col] += A[row][k] * B[k][col];
                }
        }

        public static void transpose_matr(double[][] A, out double[][] A_t)
        {
            int m_size = A.Length;
            A_t = new double[m_size][];
            for (int row = 0; row < m_size; row++)
                A_t[row] = new double[A[row].Length];


            for (int row = 0; row < m_size; row++)
                for (int col = 0; col < m_size; col++)
                    A_t[row][col] = A[col][row];
        }

        // Recursive computation of eigenvalues and eigen-vectors
        // Compute eigenvalue[i] and eigen_vecs[i], where i = eig_count
        public static void GetEigenvalsEigenvecs(double[][] matrix, ref double[] eigenvalues, ref double[][] eigen_vecs, int eig_count, double[][] matrix_i = null)
        {
            Random rnd = new Random();
            int m_size = matrix.Length;
            double[] vec = new double[m_size];
            for (int i = 0; i < m_size; i++)
                vec[i] = rnd.NextDouble();
            if (matrix_i == null)
            {
                eigenvalues = new double[m_size];
                eigen_vecs = new double[eigenvalues.Length][];
                matrix_i = (double[][])(matrix.Clone());
            }

            double[] m; m = new double[m_size];
            double[] m_temp; m_temp = new double[m_size];

            double lambda_old = 0;

            // Power Iteration algoritm for finding eigenvalues of the symmetric matrix A^t*A  
            // The solution is obtained as  (A^t*A)^\infinity * arbitrary vector

            int index = 0; bool is_eval = false;
            while (is_eval == false)
            {
                // m - will be eigenvector
                for (int row = 0; row < m_size; row++)
                    m[row] = 0f;

                for (int row = 0; row < m_size; row++)
                {
                    for (int col = 0; col < m_size; col++)
                        m[row] += matrix[row][col] * vec[col];
                }

                for (int col = 0; col < m_size; col++)
                    vec[col] = m[col];

                if (index > 0)
                {
                    // finish compute eigenvalue if lambda almost const
                    double lambda = (index > 0) ? (m[0] / m_temp[0]) : m[0];
                    is_eval = (Math.Abs(lambda - lambda_old) <= 10e-10) ? true : false;

                    eigenvalues[eig_count] = lambda;
                    lambda_old = lambda;
                }

                for (int row = 0; row < m_size; row++)
                    m_temp[row] = m[row];

                index++;
            }

            double[][] matrix_new = new double[m_size][];

            if (m_size > 1)
            {
                double[][] M_target;
                M_target = new double[m_size][];

                for (int row = 0; row < m_size; row++)
                    M_target[row] = new double[m_size];

                // M_target is (A^t * A - eigval*I)
                for (int row = 0; row < m_size; row++)
                    for (int col = 0; col < m_size; col++)
                        M_target[row][col] = (row == col) ?
                        (matrix[row][col] - eigenvalues[eig_count]) : matrix[row][col];

                // Get eigen_vecs[i]
                double[] eigen_vec;
                GaussJordanElimination(M_target, out eigen_vec);

                //  Matrix H - Conjugate for matrix A^t*A
                double[][] H;
                ConjugateFor_M_target(eigen_vec, out H);

                double[][] H_matrix_prod;
                prod_matrix(H, matrix, out H_matrix_prod);

                // inverse matrix for H: inv_H
                double[][] inv_H;
                InverseConjugateFor_M_target(eigen_vec, out inv_H);

                double[][] inv_H_matrix_prod;

                // Here, we get inv_H_matrix_prod = H * M * H^-1
                prod_matrix(H_matrix_prod, inv_H, out inv_H_matrix_prod);

                // matrix_new = H * M * H^-1 is equalent to  A_t_x_A without first row and first col
                ReduceMatrix(inv_H_matrix_prod, out matrix_new, m_size - 1);

            }

            if (m_size <= 1)
            {
                for (index = 0; index < eigenvalues.Length; index++)
                {
                    double lambda = eigenvalues[index];

                    // M_target is (A^t * A - eigval*I)
                    double[][] M_target;
                    M_target = new double[matrix_i.Length][];

                    for (int row = 0; row < matrix_i.Length; row++)
                        M_target[row] = new double[matrix_i.Length];

                    int mi_size = matrix_i.Length;

                    for (int row = 0; row < mi_size; row++)
                        for (int col = 0; col < mi_size; col++)
                            M_target[row][col] = (row == col) ?
                        (matrix_i[row][col] - lambda) : matrix_i[row][col];

                    //	eigen_vecs = new double[matrix_i.Length][];
                    GaussJordanElimination(M_target, out eigen_vecs[index]);

                    // Normalize eigen vectors
                    double eigsum_sq = 0;

                    for (int v = 0; v < eigen_vecs[index].Length; v++)
                        eigsum_sq += Math.Pow(eigen_vecs[index][v], 2.0);

                    for (int v = 0; v < eigen_vecs[index].Length; v++)
                        eigen_vecs[index][v] /= Math.Sqrt(eigsum_sq);

                    // Essentially eigenvalues[index] should be positive for symmetric matrix,
                    // However, negative values or very small values 
                    // may occur due to the accuracy of the calculations.
                    if (eigenvalues[index] < 10e-5)
                        eigenvalues[index] = 0;

                    eigenvalues[index] = Math.Sqrt(eigenvalues[index]);
                }
                return;
            }
            if (matrix_new != null)
                GetEigenvalsEigenvecs(matrix_new, ref eigenvalues, ref eigen_vecs, eig_count + 1, matrix_i);
        }

        // Discard first row and first column
        public static void ReduceMatrix(double[][] matrix, out double[][] reduced_matr, int new_size)
        {
            reduced_matr = new double[new_size][];
            if (new_size > 1)
            {
                int index_d = matrix.Length - new_size;
                int row = index_d, row_n = 0;
                while (row < matrix.Length)
                {
                    reduced_matr[row_n] = new double[new_size];
                    int col = index_d, col_n = 0;
                    while (col < matrix.Length)
                        reduced_matr[row_n][col_n++] = matrix[row][col++];

                    row++; row_n++;
                }
            }

            else if (new_size == 1)
            {
                reduced_matr[0] = new double[new_size];
                reduced_matr[0][0] = matrix[1][1];
            }
        }

        public static void ConjugateFor_M_target(double[] eigen_vec, out double[][] h_matrix)
        {
            h_matrix = new double[eigen_vec.Length][];

            for (int row = 0; row < eigen_vec.Length; row++)
                h_matrix[row] = new double[eigen_vec.Length];

            h_matrix[0][0] = 1.0 / eigen_vec[0];

            for (int row = 1; row < eigen_vec.Length; row++)
                h_matrix[row][0] = -eigen_vec[row] / eigen_vec[0];

            for (int row = 1; row < eigen_vec.Length; row++)
                h_matrix[row][row] = 1;
        }

        public static void InverseConjugateFor_M_target(double[] eigen_vec, out double[][] ih_matrix)
        {
            ih_matrix = new double[eigen_vec.Length][];
            for (int row = 0; row < eigen_vec.Length; row++)
                ih_matrix[row] = new double[eigen_vec.Length];

            ih_matrix[0][0] = eigen_vec[0];
            for (int row = 1; row < eigen_vec.Length; row++)
                ih_matrix[row][0] = -eigen_vec[row];

            for (int row = 1; row < eigen_vec.Length; row++)
                ih_matrix[row][row] = 1;
        }


        // Gauss Jordan elimination algorithm --> reduce the matrix to  row echelon form.
        // 1 0 0 v1 
        // 0 1 0 v2
        // 0 0 1 v3
        // 0 0 0  0
        // The solution of Av 0 is (-v1, -v2, -v3, 1)^t
        // The eigenvector v is optaibed up to factor, so the last element may be set to 1

        // Get eigen_vector for given M = A^t*A*v - \lambda*v
        public static void GaussJordanElimination(double[][] M, out double[] eigen_vec)
        {
            eigen_vec = new double[M.Length];
            for (int s = 0; s < M.Length - 1; s++)
            {
                double diag_elem = M[s][s];
                if (diag_elem != 0 && diag_elem != 1)
                {
                    for (int col = s; col < M[s].Length; col++)
                        M[s][col] /= diag_elem;
                }


                // Move the column with element M[s][s] to the end of matrix
                if (diag_elem == 0)
                {
                    for (int col = s; col < M[s].Length; col++)
                    {
                        double tmp = M[s][col];
                        M[s][col] = M[s + 1][col];
                        M[s + 1][col] = tmp;
                    }
                }

                // GaussJordan elimination process
                for (int row = 0; row < M.Length; row++)
                {
                    double element = M[row][s];
                    if (row != s)
                    {
                        // Eliminate element M[row][s]. Subtract the line 's' from line 'row != s' 
                        for (int col = s; col < M[row].Length; col++)
                            M[row][col] = M[row][col] - M[s][col] * element;
                    }
                }
            }

            int r = 0;
            while (r < M.Length)
                eigen_vec[r] = (-M[r++][M.Length - 1]);

            eigen_vec[eigen_vec.Length - 1] = 1;
        }

        public static void VerifyFactorisition(double[][] u, double[][] s,
            double[][] v)
        {

            double[][] prod_US, prod_USVt, v_transp;

            prod_matrix(u, s, out prod_US);
            transpose_matr(v, out v_transp);
            prod_matrix(prod_US, v_transp, out prod_USVt);
        }

        public static void VerifyPseudoinverse(double[][] A, double[][] A_pesudoinv)
        {

            double[][] prod;

            // Product should be Identity matrix or with several zeros on the diagaonal  
            prod_matrix(A, A_pesudoinv, out prod);
        }


        public static int GetRank(double[][] s)
        {
            int rank = 0;

            for (int row = 0; row < s.Length; row++)
                if (s[row][row] != 0)
                {
                    rank++;
                }

            return rank;
        }

        public static void GetPseudoinverse(double[][] s, double[][] u, double[][] v, out double[][] pseudoinv)
        {

            double[][] s_inv, prod_VS_inv, u_transp;
            InverseDiagMatrix(s, out s_inv);

            prod_matrix(v, s_inv, out prod_VS_inv);

            transpose_matr(u, out u_transp);
            prod_matrix(prod_VS_inv, u_transp, out pseudoinv);

            int rank = GetRank(s);
        }
    }
}
