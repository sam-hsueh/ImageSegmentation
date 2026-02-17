using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Utilities
{
    public class AlphaShapeCalculator
    {
        public double Alpha { get; set; }
        public bool CloseShape { get; set; }
        //public double Radius => 1 / Alpha;
        public double Radius => Alpha;

        private List<Tuple<int, int>> resultingEdges = new List<Tuple<int, int>>();
        private List<Point> resultingVertices = new List<Point>();
        private Point[] points;

        public Shape CalculateShape(Point[] points)
        {
            SetData(points);
            CalculateShape();
            if (CloseShape)
            {
                CloseShapeImpl();
            }
            return GetShape();
        }

        private void CloseShapeImpl()
        {
            var vertexCounter = CountVertices();
            var vertexIndices = vertexCounter.GetIndicesByCount(1);
            AddClosingEdges(vertexIndices);
        }

        private void AddClosingEdges(int[] vertexIndices)
        {
            foreach (var vertexIndex in vertexIndices)
            {
                var nearestPendingVertex = GetNearestPendingVertex(vertexIndices, vertexIndex);
                AddEdge(resultingVertices[vertexIndex], resultingVertices[nearestPendingVertex]);
            }
        }

        private void SetData(Point[] points)
        {
            resultingEdges.Clear();
            resultingVertices.Clear();
            this.points = points;
        }

        private void CalculateShape()
        {
            foreach (var point in points)
            {
                ProcessPoint(point);
            }
        }

        private VertexCounter CountVertices()
        {

            VertexCounter counter = new VertexCounter();

            foreach (var edge in resultingEdges)
            {
                counter.IncreaseForIndex(edge.Item1);
                counter.IncreaseForIndex(edge.Item2);
            }

            return counter;
        }

        private int GetNearestPendingVertex(int[] vertices, int vertexIndex)
        {
            var vertexPoint = GetVertex(vertexIndex);
            var vertexIndicesWithDistance =
                vertices.Where(v => v != vertexIndex).Select(v => new { Index = v, Distance = resultingVertices[v].DistanceTo(vertexPoint) });
            return vertexIndicesWithDistance.Aggregate((a, b) => a.Distance < b.Distance ? a : b).Index;
        }

        private Point GetVertex(int vertexIndex)
        {
            return resultingVertices[vertexIndex];
        }

        private void ProcessPoint(Point point)
        {
            foreach (var otherPoint in NearbyPoints(point))
            {
                Tuple<Point, Point> alphaDiskCenters = CalculateAlphaDiskCenters(point, otherPoint);

                if (!DoOtherPointsFallWithinDisk(alphaDiskCenters.Item1, point, otherPoint)
                    || !DoOtherPointsFallWithinDisk(alphaDiskCenters.Item2, point, otherPoint))
                {
                    AddEdge(point, otherPoint);
                }
            }
        }

        private bool DoOtherPointsFallWithinDisk(Point center, Point p1, Point p2)
        {
            return NearbyPoints(center).Count(p => p != p1 && p != p2) > 0;
        }

        private void AddEdge(Point p1, Point p2)
        {
            int indexP1;
            int indexP2;

            indexP1 = AddVertex(p1);
            indexP2 = AddVertex(p2);

            AddEdge(indexP1, indexP2);
        }

        private void AddEdge(int indexP1, int indexP2)
        {
            if (!resultingEdges.Contains(new Tuple<int, int>(indexP1, indexP2))
                && !resultingEdges.Contains(new Tuple<int, int>(indexP2, indexP1)))
                resultingEdges.Add(new Tuple<int, int>(indexP1, indexP2));
        }

        private int AddVertex(Point p)
        {
            int index;
            if (!resultingVertices.Contains(p))
            {
                resultingVertices.Add(p);
            }
            index = resultingVertices.IndexOf(p);
            return index;
        }

        private Point[] NearbyPoints(Point point)
        {
            var nearbyPoints = points.Where(p => p.DistanceTo(point) <= Radius && p != point).ToArray();
            return nearbyPoints;
        }

        private Tuple<Point, Point> CalculateAlphaDiskCenters(Point p1, Point p2)
        {
            double distanceBetweenPoints = p1.DistanceTo(p2);
            double distanceFromConnectionLine = Math.Sqrt(Radius * Radius - distanceBetweenPoints * distanceBetweenPoints / 4);

            Point centerOfConnectionLine = p1.CenterTo(p2);
            Point vector = p1.VectorTo(p2);

            return GetAlphaDiskCenters(vector, centerOfConnectionLine, distanceFromConnectionLine);
        }

        private static Tuple<Point, Point> GetAlphaDiskCenters(Point vector, Point center, double distanceFromConnectionLine)
        {
            Point normalVector = new Point(vector.Y, -vector.X);
            return
                new Tuple<Point, Point>(
                    new Point(center.X + normalVector.X * distanceFromConnectionLine,
                        center.Y + normalVector.Y * distanceFromConnectionLine),
                    new Point(center.X - normalVector.X * distanceFromConnectionLine,
                        center.Y - normalVector.Y * distanceFromConnectionLine));
        }

        private Shape GetShape()
        {
            return new Shape(resultingVertices.ToArray(), resultingEdges.ToArray());
        }
    }
    public struct Point : IEquatable<Point>
    {
        public bool Equals(Point other)
        {
            return X.Equals(other.X) && Y.Equals(other.Y);
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            return obj is Point && Equals((Point)obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (X.GetHashCode() * 397) ^ Y.GetHashCode();
            }
        }

        public static bool operator ==(Point left, Point right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Point left, Point right)
        {
            return !left.Equals(right);
        }

        public double X { get; private set; }
        public double Y { get; private set; }

        public Point(double x, double y)
        {
            X = x;
            Y = y;
        }

        public override string ToString()
        {
            return $"X={X}; Y={Y}";
        }

        public double DistanceTo(Point p)
        {
            return Math.Sqrt((X - p.X) * (X - p.X) + (Y - p.Y) * (Y - p.Y));
        }

        public Point CenterTo(Point p)
        {
            return new Point((X + p.X) / 2, (Y + p.Y) / 2);
        }

        public Point VectorTo(Point p)
        {
            double d = DistanceTo(p);
            return new Point((p.X - X) / d,
                (p.Y - Y) / d);
        }
    }
    public class Shape
    {
        public Shape(Point[] vertices, Tuple<int, int>[] edges)
        {
            Vertices = vertices;
            Edges = edges;
        }

        public Point[] Vertices { get; private set; }
        public Tuple<int, int>[] Edges { get; private set; }
    }
    internal class VertexCounter
    {
        Dictionary<int, int> vertexCounts = new Dictionary<int, int>();

        public void IncreaseForIndex(int index)
        {
            if (!vertexCounts.ContainsKey(index))
            {
                vertexCounts.Add(index, 0);
            }
            vertexCounts[index]++;
        }

        public int[] GetIndicesByCount(int count)
        {
            return vertexCounts.Where(kvp => kvp.Value == count).Select(kvp => kvp.Key).ToArray();
        }
    }
}
