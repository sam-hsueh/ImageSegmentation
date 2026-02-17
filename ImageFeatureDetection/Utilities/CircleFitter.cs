using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;

namespace Utilities
{
    public class CircleFitter
    {
        /** Current circle center. */
        private PointF center;

        /** Current circle radius. */
        private double rHat;

        /** Circular ring sample points. */
        private PointF[] points;

        /** Current cost function value. */
        private double J;

        /** Current cost function gradient. */
        private double dJdx;
        private double dJdy;
        public static void LoadFile(string FilePath)
        {
            try
            {

                StreamReader br = new StreamReader(FilePath);
                // read the points, ignoring blank lines and comment lines
                List<PointF> list = new List<PointF>();
                int l = 0;
                for (string line = br.ReadLine(); line != null; line = br.ReadLine())
                {
                    ++l;
                    line = line.Trim();
                    if (line.Length > 0 && !line.StartsWith("#"))
                    {
                        // this is a data line, we expect two numerical fields
                        string[] fields = line.Split(new char[] { 's' });
                        if (fields.Length != 2)
                        {
                            throw new Exception("syntax error at line " + l + ": " + line
                                                     + "(expected two fields, found"
                                                     + fields.Length + ")");
                        }

                        // parse the fields and add the point to the list
                        list.Add(new PointF(float.Parse(fields[0]), float.Parse(fields[1])));

                    }
                }
                PointF[] points = list.ToArray();

                // fit a circle to the test points
                CircleFitter fitter = new CircleFitter();
                fitter.initialize(points);
                /*
                System.out.println("initial circle: "
                                   + format.format(fitter.getCenter().X)
                                   + " " + format.format(fitter.getCenter().Y)
                                   + " " + format.format(fitter.getRadius()));

                // minimize the residuals
                int iter = fitter.minimize(100, 0.1, 1.0e-12);
                System.out.println("converged after " + iter + " iterations");
                System.out.println("final circle: "
                                   + format.format(fitter.getCenter().X)
                                   + " " + format.format(fitter.getCenter().Y)
                                   + " " + format.format(fitter.getRadius()));
                                   */

            }
            catch (IOException ioe)
            {
            }
        }

        /** Build a new instance with a default current circle.
         */
        public CircleFitter()
        {
            PointF center = new PointF(0.0f, 0.0f);
            double rHat = 1.0f;
            PointF[] points = new PointF[1000];
            Random rnd = new Random();
            double r = 450;
            double cx = 500, cy = 600;
            for (int i = 0; i < 1000; i++)
            {
                double ang = i / 572.5978;
                double tr = r + rnd.Next(60) - 30.0;
                points[i] = new PointF((float)(tr * Math.Sin(ang) + cx), (float)(tr * Math.Cos(ang) + cy));
            }
            this.points = points;
        }


        /** Initialize an approximate circle based on all triplets.
         * @param points circular ring sample points
         * @exception Exception if all points are aligned
         */
        public void initialize(PointF[] ps = null)
        {

            // store the points array
            if (ps != null)
                points = ps;

            // analyze all possible points triplets
            center.X = 0.0f;
            center.Y = 0.0f;
            int n = 0;
            for (int i = 0; i < points.Length - 2; ++i)
            {
                PointF p1 = points[i];
                for (int j = i + 1; j < points.Length - 1; ++j)
                {
                    PointF p2 = points[j];
                    for (int k = j + 1; k < points.Length; ++k)
                    {
                        PointF p3 = points[k];

                        // compute the triangle circumcenter
                        PointF cc = circumcenter(p1, p2, p3);
                        if (cc != null)
                        {
                            // the points are not aligned, we have a circumcenter
                            ++n;
                            center.X += cc.X;
                            center.Y += cc.Y;
                        }
                    }
                }
            }

            if (n == 0)
            {
                throw new Exception("all points are aligned");
            }

            // initialize using the circumcenters average
            center.X /= n;
            center.Y /= n;
            updateRadius();
        }

        /** Update the circle radius.
         */
        private void updateRadius()
        {
            double rHat = 0;
            for (int i = 0; i < points.Length; ++i)
            {
                double dx = points[i].X - center.X;
                double dy = points[i].Y - center.Y;
                rHat += Math.Sqrt(dx * dx + dy * dy);
            }
            rHat /= points.Length;
        }

        /** Compute the circumcenter of three points.
         * @param pI first point
         * @param pJ second point
         * @param pK third point
         * @return circumcenter of pI, pJ and pK or null if the points are aligned
         */
        private PointF circumcenter(PointF pI, PointF pJ, PointF pK)
        {

            // some temporary variables
            PointF dIJ = new PointF(pJ.X - pI.X, pJ.Y - pI.Y);
            PointF dJK = new PointF(pK.X - pJ.X, pK.Y - pJ.Y);
            PointF dKI = new PointF(pI.X - pK.X, pI.Y - pK.Y);
            double sqI = pI.X * pI.X + pI.Y * pI.Y;
            double sqJ = pJ.X * pJ.X + pJ.Y * pJ.Y;
            double sqK = pK.X * pK.X + pK.Y * pK.Y;

            // determinant of the linear system: 0 for aligned points
            double det = dJK.X * dIJ.Y - dIJ.X * dJK.Y;
            if (Math.Abs(det) < 1.0e-10)
            {
                // points are almost aligned, we cannot compute the circumcenter
                return new PointF();
            }

            // beware, there is a minus sign on Y coordinate!
            return new PointF((float)((sqI * dJK.Y + sqJ * dKI.Y + sqK * dIJ.Y) / (2 * det)), -(float)((sqI * dJK.X + sqJ * dKI.X + sqK * dIJ.X) / (2 * det)));

        }

        /** Minimize the distance residuals between the points and the circle.
         * <p>We use a non-linear conjugate gradient method with the Polak and
         * Ribiere coefficient for the computation of the search direction. The
         * inner minimization along the search direction is performed using a
         * few Newton steps. It is worthless to spend too much time on this inner
         * minimization, so the convergence threshold can be rather large.</p>
         * @param maxIter maximal iterations number on the inner loop (cumulated
         * across outer loop iterations)
         * @param innerThreshold inner loop threshold, as a relative difference on
         * the cost function value between the two last iterations
         * @param outerThreshold outer loop threshold, as a relative difference on
         * the cost function value between the two last iterations
         * @return number of inner loop iterations performed (cumulated
         * across outer loop iterations)
         * @exception Exception if we come accross a singularity or if
         * we exceed the maximal number of iterations
         */
        public int minimize(int iterMax, double innerThreshold, double outerThreshold)
        {

            computeCost();
            if (J < 1.0e-10 || Math.Sqrt(dJdx * dJdx + dJdy * dJdy) < 1.0e-10)
            {
                // we consider we are already at a local minimum
                return 0;
            }

            double previousJ = J;
            double previousU = 0.0, previousV = 0.0;
            double previousDJdx = 0.0, previousDJdy = 0.0;
            for (int iterations = 0; iterations < iterMax;)
            {

                // search direction
                double u = -dJdx;
                double v = -dJdy;
                if (iterations != 0)
                {
                    // Polak-Ribiere coefficient
                    double beta =
                      (dJdx * (dJdx - previousDJdx) + dJdy * (dJdy - previousDJdy))
                    / (previousDJdx * previousDJdx + previousDJdy * previousDJdy);
                    u += beta * previousU;
                    v += beta * previousV;
                }
                previousDJdx = dJdx;
                previousDJdy = dJdy;
                previousU = u;
                previousV = v;

                // rough minimization along the search direction
                double innerJ;
                do
                {
                    innerJ = J;
                    double lambda = newtonStep(u, v);
                    center.X += (float)(lambda * u);
                    center.Y += (float)(lambda * v);
                    updateRadius();
                    computeCost();
                } while (++iterations < iterMax
                         && Math.Abs(J - innerJ) / J > innerThreshold);

                // global convergence test
                if (Math.Abs(J - previousJ) / J < outerThreshold)
                {
                    return iterations;
                }
                previousJ = J;

            }

            throw new Exception("unable to converge after " + iterMax + " iterations");
        }

        /** Compute the cost function and its gradient.
         * <p>The results are stored as instance attributes.</p>
         */
        private void computeCost()
        {
            J = 0;
            dJdx = 0;
            dJdy = 0;
            for (int i = 0; i < points.Length; ++i)
            {
                double dx = points[i].X - center.X;
                double dy = points[i].Y - center.Y;
                double di = Math.Sqrt(dx * dx + dy * dy);
                if (di < 1.0e-10)
                {
                    throw new Exception("cost singularity:"
                                             + " point at the circle center");
                }
                double dr = di - rHat;
                double ratio = dr / di;
                J += dr * (di + rHat);
                dJdx += dx * ratio;
                dJdy += dy * ratio;
            }
            dJdx *= 2.0;
            dJdy *= 2.0;
        }

        /** Compute the Length of the Newton step in the search direction.
         * @param u abscissa of the search direction
         * @param v ordinate of the search direction
         * @return value of the step along the search direction
         */
        private double newtonStep(double u, double v)
        {

            // compute the first and second derivatives of the cost
            // along the specified search direction
            double sum1 = 0, sum2 = 0, sumFac = 0, sumFac2R = 0;
            for (int i = 0; i < points.Length; ++i)
            {
                double dx = center.X - points[i].X;
                double dy = center.Y - points[i].Y;
                double di = Math.Sqrt(dx * dx + dy * dy);
                double coeff1 = (dx * u + dy * v) / di;
                double coeff2 = di - rHat;
                sum1 += coeff1 * coeff2;
                sum2 += coeff2 / di;
                sumFac += coeff1;
                sumFac2R += coeff1 * coeff1 / di;
            }

            // step Length attempting to nullify the first derivative
            return -sum1 / ((u * u + v * v) * sum2
                            - sumFac * sumFac / points.Length
                            + rHat * sumFac2R);
        }

        /** Get the circle center.
         * @return circle center
         */
        public PointF getCenter()
        {
            return center;
        }

        /** Get the circle radius.
         * @return circle radius
         */
        public double getRadius()
        {
            return rHat;
        }
    }
}
