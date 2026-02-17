using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Shapes;

namespace Utilities
{
    public class GrahamScan
    {
        private Point basePoint;
        public List<Point> Run(List<Point> points)
        {
            List<Point> outPoints = new List<Point>(); 
            List<Point> grahamPoints = new List<Point>(points);
            List<Point> ConvexPoints = new List<Point>();
            int baseIndex = getminYPointIndex(grahamPoints);
            basePoint = grahamPoints[baseIndex];
            grahamPoints.RemoveAt(baseIndex);
            ConvexPoints.Add(basePoint);
            if (grahamPoints.Count == 0) { outPoints.Add(basePoint); return null; }
            if (grahamPoints.Count == 1)
            {
                outPoints.Add(basePoint);
                if (grahamPoints[0] != basePoint)
                    outPoints.Add(grahamPoints[0]);
                return null;
            }
            grahamPoints.Sort(CompareByAngle);
            ConvexPoints.Add(grahamPoints[0]);
            grahamPoints.Remove(grahamPoints[0]);

            while (grahamPoints.Count > 0)
            {
                if (ConvexPoints.Count == 1)
                {
                    // if(!Co//////////////////////nvexPoints.Co////////////ntains(grahamPoints[0]))
                    ConvexPoints.Add(grahamPoints[0]);
                    grahamPoints.Remove(grahamPoints[0]);
                    continue;
                }
                Point Ptop = ConvexPoints[ConvexPoints.Count - 1];
                Point Pprev = ConvexPoints[ConvexPoints.Count - 2];

                Point Pi = grahamPoints[0];
                if (CheckTurn(new Line(Pprev, Ptop), Pi) == Enums.TurnType.Left)
                {
                    ConvexPoints.Add(Pi);
                    grahamPoints.Remove(Pi);
                }
                else
                    ConvexPoints.RemoveAt(ConvexPoints.Count - 1);
            }
            outPoints = new List<Point>(ConvexPoints);
            return outPoints;
        }
        public int getminYPointIndex(List<Point> points)
        {
            int index = -1;
            double y = 100000000000000000;
            for (int i = 0; i < points.Count; i++)
            {
                if (points[i].Y <= y)
                {
                    y = points[i].Y;
                    index = i;
                }
            }
            return index;
        }
        public double CalculateAngel(Point vec1, Point vec2)
        {
            double cross = CrossProduct(vec1, vec2);
            double dot = DotProduct(vec1, vec2);
            double seta = Math.Atan2(cross, dot) * 180 / Math.PI;
            if (seta < 0)
                seta += 360;
            return seta;
        }
        private int CompareByAngle(Point point1, Point point2)
        {
            Point supportPoint = new Point(basePoint.X + 7, basePoint.Y);
            Point supportVector = new Point(supportPoint.X - basePoint.X, supportPoint.Y - basePoint.Y);
            Point V1 = new Point(point1.X - basePoint.X, point1.Y - basePoint.Y); 
            Point V2 = new Point(point2.X - basePoint.X, point2.Y - basePoint.Y);
            double seta1 = CalculateAngel(supportVector, V1);
            double seta2 = CalculateAngel(supportVector, V2);
            int res = 0;
            if (seta1 > seta2)
            { res = 1; }
            else if (seta1 < seta2)
            { res = -1; }
            else if (seta1 == seta2)
            {
                double D1 = EuclideanDistance(basePoint, point1);
                double D2 = EuclideanDistance(basePoint, point2);
                if (D1 >= D2) res = 1;
                else res = -1;
            }
            return res;
        }
        public static Enums.TurnType CheckTurn(Point vector1, Point vector2)
        {
            double result = CrossProduct(vector1, vector2);
            if (result < 0) return Enums.TurnType.Right;
            else if (result > 0) return Enums.TurnType.Left;
            else return Enums.TurnType.Colinear;
        }

        public static double CrossProduct(Point a, Point b)
        {
            return a.X * b.Y - a.Y * b.X;
        }
        public static double DotProduct(Point a, Point b)
        {
            return a.X * b.X + a.Y * b.Y;
        }
        public static Enums.TurnType CheckTurn(Line l, Point p)
        {
            Point a = new Point(l.End.X - l.Start.X, l.End.Y - l.Start.Y);
            Point b = new Point(l.End.X - p.X, l.End.Y - p.Y);
            return CheckTurn(a, b);
        }
        public static Point GetVector(Line l)
        {
            return new Point(l.End.X-l.Start.X,l.End.Y-l.Start.Y);
        }
        public static double EuclideanDistance(Point p, Point b)
        {
            double dY = p.Y - b.Y;
            double dX = p.X - b.X;
            double res = Math.Sqrt((dY * dY) + (dX * dX));
            return res;
        }
    }
    /// <summary>
    /// The primary Line structure to be used in the CG project.
    /// </summary>
    public class Line : ICloneable
    {
        /// <summary>
        /// Creates a line structure that has the specified start/end.
        /// </summary>
        /// <param name="start">The start point.</param>
        /// <param name="end">The end point.</param>
        public Line(Point start, Point end)
        {
            this.Start = start;
            this.End = end;
        }

        /// <summary>
        /// Creates a line structure that has the specified start/end.
        /// </summary>
        /// <param name="X1">The X value for the start point.</param>
        /// <param name="Y1">The Y value for the start point.</param>
        /// <param name="X2">The X value for the end point.</param>
        /// <param name="Y2">The Y value for the end point.</param>
        public Line(double X1, double Y1, double X2, double Y2)
            : this(new Point((int)X1, (int)Y1), new Point((int)X2, (int)Y2))
        {
        }

        /// <summary>
        /// Gets or sets the start point.
        /// </summary>
        public Point Start
        {
            get;
            set;
        }

        /// <summary>
        /// Gets or sets the end point.
        /// </summary>
        public Point End
        {
            get;
            set;
        }
        /// <summary>
        /// Instantiate Line
        /// </summary>
        /// <returns>new instance of Line</returns>
        public object Clone()
        {
            return new Line(new Point(Start.X,Start.Y), new Point(End.X,End.Y));
        }
    }
    public class Enums
    {
        public enum TurnType
        {
            Left,
            Right,
            Colinear
        }
        public enum PointInPolygon
        {
            Inside,
            Outside,
            OnEdge
        }
        public enum PolygonOrder
        {
            Clockwise,
            CounterClockWise
        }
        public enum EventType { Start, End, Intersection }
    }
}
