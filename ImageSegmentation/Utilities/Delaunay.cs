using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace Utilities
{
    internal class Delaunay
    {
        bool Circumcircle(PointF p0, PointF p1, PointF p2, PointF center, double radius)
        {
            double dA, dB, dC, aux1, aux2, div;

            dA = p0.X * p0.X + p0.Y * p0.Y;
            dB = p1.X * p1.X + p1.Y * p1.Y;
            dC = p2.X * p2.X + p2.Y * p2.Y;

            aux1 = (dA * (p2.Y - p1.Y) + dB * (p0.Y - p2.Y) + dC * (p1.Y - p0.Y));
            aux2 = -(dA * (p2.X - p1.X) + dB * (p0.X - p2.X) + dC * (p1.X - p0.X));
            div = (2 * (p0.X * (p2.Y - p1.Y) + p1.X * (p0.Y - p2.Y) + p2.X * (p1.Y - p0.Y)));

            if (div == 0)///三点一线
            {
                return false;
            }

            center.X = (float)(aux1 / div);
            center.Y = (float)(aux2 / div);

            radius = Math.Sqrt((center.X - p0.X) * (center.X - p0.X) + (center.Y - p0.Y) * (center.Y - p0.Y));

            return true;
        }
    }
    public class DelaunayTriangulation
    {
        public const int MaxVertices = 500;
        public const int MaxTriangles = 1000;
        public dVertex[] Vertex = new dVertex[MaxVertices];
        public dTriangle[] Triangle = new dTriangle[MaxTriangles];
        private bool InCircle(long xp, long yp, long x1, long y1, long x2, long y2, long x3, long y3, double xc, double yc, double r)
        {
            double eps;
            double m1;
            double m2;
            double mx1;
            double mx2;
            double my1;
            double my2;
            double dx;
            double dy;
            double rsqr;
            double drsqr;
            eps = 0.000000001;
            if (Math.Abs(y1 - y2) < eps && Math.Abs(y2 - y3) < eps)
            {
                return false;
            }
            if (Math.Abs(y2 - y1) < eps)
            {
                m2 = (-(Convert.ToDouble(x3) - Convert.ToDouble(x2)) / (Convert.ToDouble(y3) - Convert.ToDouble(y2)));
                mx2 = Convert.ToDouble((x2 + x3) / 2.0);
                my2 = Convert.ToDouble((y2 + y3) / 2.0);
                xc = Convert.ToDouble((x2 + x1) / 2.0);
                yc = Convert.ToDouble(m2 * (xc - mx2) + my2);
            }
            else if (Math.Abs(y3 - y2) < eps)
            {
                m1 = (-(Convert.ToDouble(x2) - Convert.ToDouble(x1)) / (Convert.ToDouble(y2) - Convert.ToDouble(y1)));
                mx1 = Convert.ToDouble((x1 + x2) / 2.0);
                my1 = Convert.ToDouble((y1 + y2) / 2.0);
                xc = Convert.ToDouble((x3 + x2) / 2.0);
                yc = Convert.ToDouble(m1 * (xc - mx1) + my1);
            }
            else
            {
                m1 = (-(Convert.ToDouble(x2) - Convert.ToDouble(x1)) / (Convert.ToDouble(y2) - Convert.ToDouble(y1)));
                m2 = (-(Convert.ToDouble(x3) - Convert.ToDouble(x2)) / (Convert.ToDouble(y3) - Convert.ToDouble(y2)));
                mx1 = Convert.ToDouble((x1 + x2) / 2.0);
                mx2 = Convert.ToDouble((x2 + x3) / 2.0);
                my1 = Convert.ToDouble((y1 + y2) / 2.0);
                my2 = Convert.ToDouble((y2 + y3) / 2.0);
                xc = Convert.ToDouble((m1 * mx1 - m2 * mx2 + my2 - my1) / (m1 - m2));
                yc = Convert.ToDouble(m1 * (xc - mx1) + my1);
            }
            dx = (Convert.ToDouble(x2) - Convert.ToDouble(xc));
            dy = (Convert.ToDouble(y2) - Convert.ToDouble(yc));
            rsqr = Convert.ToDouble(dx * dx + dy * dy);
            r = Convert.ToDouble(Math.Sqrt(rsqr));
            dx = Convert.ToDouble(xp - xc);
            dy = Convert.ToDouble(yp - yc);
            drsqr = Convert.ToDouble(dx * dx + dy * dy);
            if (drsqr <= rsqr)
            {
                return true;
            }
            return false;
        }
        private int WhichSide(long xp, long yp, long x1, long y1, long x2, long y2)
        {
            double equation;
            equation = ((Convert.ToDouble(yp) - Convert.ToDouble(y1)) * (Convert.ToDouble(x2) - Convert.ToDouble(x1))) - ((Convert.ToDouble(y2) - Convert.ToDouble(y1)) * (Convert.ToDouble(xp) - Convert.ToDouble(x1)));
            if (equation > 0)
            {
                return -1;
                //WhichSide = -1;
            }
            else if (equation == 0)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }
        public int Triangulate(int nvert)
        {
            bool[] Complete = new bool[MaxTriangles];
            long[,] Edges = new long[3, MaxTriangles * 3 + 1];
            long Nedge;
            long xmin;
            long xmax;
            long ymin;
            long ymax;
            long xmid;
            long ymid;
            double dx;
            double dy;
            double dmax;
            int i;
            int j;
            int k;
            int ntri;
            double xc = 0.0;
            double yc = 0.0;
            double r = 0.0;
            bool inc;
            xmin = Vertex[1].x;
            ymin = Vertex[1].y;
            xmax = xmin;
            ymax = ymin;
            for (i = 2; i <= nvert; i++)
            {
                if (Vertex[i].x < xmin)
                {
                    xmin = Vertex[i].x;
                }
                if (Vertex[i].x > xmax)
                {
                    xmax = Vertex[i].x;
                }
                if (Vertex[i].y < ymin)
                {
                    ymin = Vertex[i].y;
                }
                if (Vertex[i].y > ymax)
                {
                    ymax = Vertex[i].y;
                }
            }
            dx = Convert.ToDouble(xmax) - Convert.ToDouble(xmin);
            dy = Convert.ToDouble(ymax) - Convert.ToDouble(ymin);
            if (dx > dy)
            {
                dmax = dx;
            }
            else
            {
                dmax = dy;
            }
            xmid = (xmax + xmin) / 2;
            ymid = (ymax + ymin) / 2;
            Vertex[nvert + 1].x = Convert.ToInt64(xmid - 2 * dmax);
            Vertex[nvert + 1].y = Convert.ToInt64(ymid - dmax);
            Vertex[nvert + 2].x = xmid;
            Vertex[nvert + 2].y = Convert.ToInt64(ymid + 2 * dmax);
            Vertex[nvert + 3].x = Convert.ToInt64(xmid + 2 * dmax);
            Vertex[nvert + 3].y = Convert.ToInt64(ymid - dmax);
            Triangle[1].vv0 = nvert + 1;
            Triangle[1].vv1 = nvert + 2;
            Triangle[1].vv2 = nvert + 3;
            Complete[1] = false;
            ntri = 1;
            for (i = 1; i <= nvert; i++)
            {
                Nedge = 0;
                j = 0;
                do
                {
                    j = j + 1;
                    if (Complete[j] != true)
                    {
                        inc = InCircle(Vertex[i].x, Vertex[i].y, Vertex[Triangle[j].vv0].x, Vertex[Triangle[j].vv0].y, Vertex[Triangle[j].vv1].x, Vertex[Triangle[j].vv1].y, Vertex[Triangle[j].vv2].x, Vertex[Triangle[j].vv2].y, xc, yc, r);
                        if (inc)
                        {
                            Edges[1, Nedge + 1] = Triangle[j].vv0;
                            Edges[2, Nedge + 1] = Triangle[j].vv1;
                            Edges[1, Nedge + 2] = Triangle[j].vv1;
                            Edges[2, Nedge + 2] = Triangle[j].vv2;
                            Edges[1, Nedge + 3] = Triangle[j].vv2;
                            Edges[2, Nedge + 3] = Triangle[j].vv0;
                            Nedge = Nedge + 3;
                            Triangle[j].vv0 = Triangle[ntri].vv0;
                            Triangle[j].vv1 = Triangle[ntri].vv1;
                            Triangle[j].vv2 = Triangle[ntri].vv2;
                            Complete[j] = Complete[ntri];
                            j = j - 1;
                            ntri = ntri - 1;
                        }
                    }
                }
                while (j < ntri);
                for (j = 1; j <= Nedge - 1; j++)
                {
                    if (Edges[1, j] != 0 && Edges[2, j] != 0)
                    {
                        for (k = j + 1; k <= Nedge; k++)
                        {
                            if (Edges[1, k] != 0 && Edges[2, k] != 0)
                            {
                                if (Edges[1, j] == Edges[2, k])
                                {
                                    if (Edges[2, j] == Edges[1, k])
                                    {
                                        Edges[1, j] = 0;
                                        Edges[2, j] = 0;
                                        Edges[1, k] = 0;
                                        Edges[2, k] = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                for (j = 1; j <= Nedge; j++)
                {
                    if (Edges[1, j] != 0 && Edges[2, j] != 0)
                    {
                        ntri = ntri + 1;
                        Triangle[ntri].vv0 = Edges[1, j];
                        Triangle[ntri].vv1 = Edges[2, j];
                        Triangle[ntri].vv2 = i;
                        Complete[ntri] = false;
                    }
                }
            }
            i = 0;
            do
            {
                i = i + 1;
                if (Triangle[i].vv0 > nvert || Triangle[i].vv1 > nvert || Triangle[i].vv2 > nvert)
                {
                    Triangle[i].vv0 = Triangle[ntri].vv0;
                    Triangle[i].vv1 = Triangle[ntri].vv1;
                    Triangle[i].vv2 = Triangle[ntri].vv2;
                    i = i - 1;
                    ntri = ntri - 1;
                }
            }
            while (i < ntri);
            return ntri;
        }
        public static double Diameter(double Ax, double Ay, double Bx, double By, double Cx, double Cy)
        {
            double x, y;
            double a = Ax;
            double b = Bx;
            double c = Cx;
            double m = Ay;
            double n = By;
            double k = Cy;
            double A = a * b * b + a * n * n + a * a * c - b * b * c + m * m * c - n * n * c - a * c * c - a * k * k - a * a * b + b * c * c - m * m * b + b * k * k;
            double B = a * n + m * c + k * b - n * c - a * k - b * m;
            y = A / B / 2;
            double AA = b * b * m + m * n * n + a * a * k - b * b * k + m * m * k - n * n * k - c * c * m - m * k * k - a * a * n + c * c * n - m * m * n + k * k * n;
            double BB = b * m + a * k + c * n - b * k - c * m - a * n;
            x = AA / BB / 2;
            return Math.Sqrt((Ax - x) * (Ax - x) + (Ay - y) * (Ay - y));
        }
        public static double MaxEdge(double Ax, double Ay, double Bx, double By, double Cx, double Cy)
        {
            double len1 = Math.Sqrt((Ax - Bx) * (Ax - Bx) + (Ay - By) * (Ay - By));
            double len2 = Math.Sqrt((Cx - Bx) * (Cx - Bx) + (Cy - By) * (Cy - By));
            double len3 = Math.Sqrt((Ax - Cx) * (Ax - Cx) + (Ay - Cy) * (Ay - Cy));
            double len = len1 > len2 ? len1 : len2;
            return len > len3 ? len : len3;
        }
    }
    public struct dVertex
    {
        public long x;
        public long y;
        public long z;
    }
    public struct dTriangle
    {
        public long vv0;
        public long vv1;
        public long vv2;
    }
    public struct OrTriangle
    {
        public Point2d p0;
        public Point2d p1;
        public Point2d p2;
    }
    //public struct Point2d
    //{
    //    public long X;
    //    public long Y;
    //    public Point2d(long x, long y)
    //    {
    //        this.X = x;
    //        this.Y = y;
    //    }
    //}

    public struct Triangle
    {
        public int P0Index;
        public int P1Index;
        public int P2Index;
        public int Index;
        public Triangle(int p0index, int p1index, int p2index)
        {
            this.P0Index = p0index;
            this.P1Index = p1index;
            this.P2Index = p2index;
            this.Index = -1;
        }
        public Triangle(int p0index, int p1index, int p2index, int index)
        {
            this.P0Index = p0index;
            this.P1Index = p1index;
            this.P2Index = p2index;
            this.Index = index;
        }
    }
    public struct EdgeInfo
    {
        public int P0Index;
        public int P1Index;
        public List<int> AdjTriangle;
        public bool Flag;
        public double Length;
        public int GetEdgeType()
        {
            return AdjTriangle.Count;
        }
        public bool IsValid()
        {
            return P0Index != -1;
        }
        public EdgeInfo(int d)
        {
            P0Index = -1;
            P1Index = -1;
            Flag = false;
            AdjTriangle = new List<int>();
            Length = -1;
        }
    }
    public class DelaunayMesh2d
    {
        public List<Point2d> Points;
        public List<Triangle> Faces;
        public EdgeInfo[,] Edges;
        public DelaunayMesh2d()
        {
            Points = new List<Point2d>();
            Faces = new List<Triangle>();
        }
        public int AddVertex(Point2d p)
        {
            Points.Add(p);
            return Points.Count - 1;
        }
        public int AddFace(Triangle t)
        {
            Faces.Add(t);
            return Faces.Count - 1;
        }
        public void InitEdgesInfo()
        {
            Edges = new EdgeInfo[Points.Count, Points.Count];
            for (int i = 0; i < Points.Count; i++)
            {
                for (int j = 0; j < Points.Count; j++)
                {
                    Edges[i, j] = new EdgeInfo(0);
                }
            }
            for (int i = 0; i < Faces.Count; i++)
            {
                Triangle t = Faces[i];
                SetEdge(t, i);
            }

        }
        private void SetEdge(Triangle t, int i)
        {
            if (t.P0Index < t.P1Index)
            {
                Edges[t.P0Index, t.P1Index].P0Index = t.P0Index;
                Edges[t.P0Index, t.P1Index].P1Index = t.P1Index;
                Edges[t.P0Index, t.P1Index].AdjTriangle.Add(i);
                Edges[t.P0Index, t.P1Index].Length = BallConcave.GetDistance(Points[t.P0Index], Points[t.P1Index]);
            }
            else
            {
                Edges[t.P1Index, t.P0Index].P0Index = t.P1Index;
                Edges[t.P1Index, t.P0Index].P1Index = t.P0Index;
                Edges[t.P1Index, t.P0Index].AdjTriangle.Add(i);
                Edges[t.P1Index, t.P0Index].Length = BallConcave.GetDistance(Points[t.P0Index], Points[t.P1Index]);
            }

            if (t.P1Index < t.P2Index)
            {
                Edges[t.P1Index, t.P2Index].P0Index = t.P1Index;
                Edges[t.P1Index, t.P2Index].P1Index = t.P2Index;
                Edges[t.P1Index, t.P2Index].AdjTriangle.Add(i);
                Edges[t.P1Index, t.P2Index].Length = BallConcave.GetDistance(Points[t.P1Index], Points[t.P2Index]);
            }
            else
            {
                Edges[t.P2Index, t.P1Index].P0Index = t.P2Index;
                Edges[t.P2Index, t.P1Index].P1Index = t.P1Index;
                Edges[t.P2Index, t.P1Index].AdjTriangle.Add(i);
                Edges[t.P2Index, t.P1Index].Length = BallConcave.GetDistance(Points[t.P1Index], Points[t.P2Index]);
            }

            if (t.P0Index < t.P2Index)
            {
                Edges[t.P0Index, t.P2Index].P0Index = t.P0Index;
                Edges[t.P0Index, t.P2Index].P1Index = t.P2Index;
                Edges[t.P0Index, t.P2Index].AdjTriangle.Add(i);
                Edges[t.P0Index, t.P2Index].Length = BallConcave.GetDistance(Points[t.P0Index], Points[t.P2Index]);
            }
            else
            {
                Edges[t.P2Index, t.P0Index].P0Index = t.P2Index;
                Edges[t.P2Index, t.P0Index].P1Index = t.P0Index;
                Edges[t.P2Index, t.P0Index].AdjTriangle.Add(i);
                Edges[t.P2Index, t.P0Index].Length = BallConcave.GetDistance(Points[t.P0Index], Points[t.P2Index]);
            }
        }
        public void ExecuteEdgeDecimation(double length)
        {
            Queue<EdgeInfo> queue = new Queue<EdgeInfo>();
            for (int i = 0; i < Points.Count; i++)
            {
                for (int j = 0; j < Points.Count; j++)
                {
                    if (i < j && Edges[i, j].IsValid())
                    {
                        if (Edges[i, j].GetEdgeType() == 0)
                        {
                            throw new Exception();
                        }
                        if (Edges[i, j].Length > length && Edges[i, j].GetEdgeType() == 1)
                        {
                            queue.Enqueue(Edges[i, j]);
                        }
                    }
                }
            }
            EdgeInfo[] opp1Temp = new EdgeInfo[2];
            while (queue.Count != 0)
            {
                EdgeInfo info = queue.Dequeue();
                if (info.AdjTriangle.Count != 1)
                    throw new Exception();
                int tindex = info.AdjTriangle[0];
                Triangle t = Faces[tindex];
                InitOppEdge(opp1Temp, t, info);
                SetInvalid(info.P0Index, info.P1Index);
                for (int i = 0; i < 2; i++)
                {
                    EdgeInfo e = opp1Temp[i];
                    e.AdjTriangle.Remove(tindex);
                    if (e.GetEdgeType() == 0)
                    {
                        SetInvalid(e.P0Index, e.P1Index);
                    }
                    else if (e.GetEdgeType() == 1 && e.Length > length)
                    {
                        queue.Enqueue(e);
                    }
                }
            }
        }
        public List<EdgeInfo> GetBoundaryEdges()
        {
            List<EdgeInfo> list = new List<EdgeInfo>();
            for (int i = 0; i < Points.Count; i++)
            {
                for (int j = 0; j < Points.Count; j++)
                {
                    if (i < j)
                    {
                        if (Edges[i, j].GetEdgeType() == 1)
                        {
                            list.Add(Edges[i, j]);
                        }
                    }
                }
            }
            return list;
        }
        private void SetInvalid(int i, int j)
        {
            Edges[i, j].AdjTriangle.Clear();
            Edges[i, j].Flag = true;
            Edges[i, j].P0Index = -1;
            Edges[i, j].P1Index = -1;
        }
        private void InitOppEdge(EdgeInfo[] opp1Temp, Triangle t, EdgeInfo info)
        {
            int vindex = t.P0Index + t.P1Index + t.P2Index - info.P0Index - info.P1Index;
            if (vindex < info.P0Index)
            {
                opp1Temp[0] = Edges[vindex, info.P0Index];
            }
            else
            {
                opp1Temp[0] = Edges[info.P0Index, vindex];
            }

            if (vindex < info.P1Index)
            {
                opp1Temp[1] = Edges[vindex, info.P1Index];
            }
            else
            {
                opp1Temp[1] = Edges[info.P1Index, vindex];
            }
        }
    }
    public class BallConcave
    {
        //public MainWindow main;
        struct Point2dInfo : IComparable<Point2dInfo>
        {
            public Point2d Point;
            public int Index;
            public double DistanceTo;
            public Point2dInfo(Point2d p, int i, double dis)
            {
                this.Point = p;
                this.Index = i;
                this.DistanceTo = dis;
            }
            public int CompareTo(Point2dInfo other)
            {
                return DistanceTo.CompareTo(other.DistanceTo);
            }
            public override string ToString()
            {
                return Point + "," + Index + "," + DistanceTo;
            }
        }
        public BallConcave(List<System.Drawing.Point> list1)
        {
            List<Point2d> list = new List<Point2d>();
            foreach (System.Drawing.Point p in list1)
            {
                list.Add(new Point2d(p.X, p.Y));
            }
            this.points = list;
            points.OrderBy(p => p.X).ThenBy(p => p.Y);
           // points.Sort();
            flags = new bool[points.Count];
            for (int i = 0; i < flags.Length; i++)
                flags[i] = false;
            InitDistanceMap();
            InitNearestList();
        }
        private bool[] flags;
        private List<Point2d> points;
        private double[,] distanceMap;
        private List<int>[] rNeigbourList;
        private void InitNearestList()
        {
            rNeigbourList = new List<int>[points.Count];
            for (int i = 0; i < rNeigbourList.Length; i++)
            {
                rNeigbourList[i] = GetSortedNeighbours(i);
            }
        }
        private void InitDistanceMap()
        {
            distanceMap = new double[points.Count, points.Count];
            for (int i = 0; i < points.Count; i++)
            {
                for (int j = 0; j < points.Count; j++)
                {
                    distanceMap[i, j] = GetDistance(points[i], points[j]);
                }
            }
        }
        public double GetRecomandedR()
        {
            double r = double.MinValue;
            for (int i = 0; i < points.Count; i++)
            {
                if (distanceMap[i, rNeigbourList[i][1]] > r)
                    r = distanceMap[i, rNeigbourList[i][1]];
            }
            return r;
        }
        public double GetMinEdgeLength()
        {
            double min = double.MaxValue;
            for (int i = 0; i < points.Count; i++)
            {
                for (int j = 0; j < points.Count; j++)
                {
                    if (i < j)
                    {
                        if (distanceMap[i, j] < min)
                            min = distanceMap[i, j];
                    }
                }
            }
            return min;
        }
        public List<System.Drawing.Point> GetConcave_Ball(double radius)
        {
            List<Point2d> ret = new List<Point2d>();
            List<int>[] adjs = GetInRNeighbourList(2 * radius);
            ret.Add(points[0]);
            //flags[0] = true;
            int i = 0, j = -1, prev = -1;
            while (true)
            {
                j = GetNextPoint_BallPivoting(prev, i, adjs[i], radius);
                if (j == -1)
                    break;
                Point2d p = BallConcave.GetCircleCenter(points[i], points[j], radius);
                ret.Add(points[j]);
                flags[j] = true;
                prev = i;
                i = j;
            }
            List<System.Drawing.Point> list = new List<System.Drawing.Point>();
            foreach (Point2d p in ret)
            {
                list.Add(new System.Drawing.Point((int)p.X, (int)p.Y));
            }
            return list;
        }
        public List<System.Drawing.Point> GetConcave_Edge(double radius)
        {
            List<Point2d> ret = new List<Point2d>();
            List<int>[] adjs = GetInRNeighbourList(2 * radius);
            ret.Add(points[0]);
            int i = 0, j = -1, prev = -1;
            while (true)
            {
                j = GetNextPoint_EdgePivoting(prev, i, adjs[i], radius);
                if (j == -1)
                    break;
                //Point2d p = BallConcave.GetCircleCenter(points[i], points[j], radius);
                ret.Add(points[j]);
                flags[j] = true;
                prev = i;
                i = j;
            }
            List<System.Drawing.Point> list = new List<System.Drawing.Point>();
            foreach (Point2d p in ret)
            {
                list.Add(new System.Drawing.Point((int)p.X, (int)p.Y));
            }
            return list;
        }
        private bool CheckValid(List<int>[] adjs)
        {
            for (int i = 0; i < adjs.Length; i++)
            {
                if (adjs[i].Count < 2)
                {
                    return false;
                }
            }
            return true;
        }
        public bool CompareAngel(Point2d a, Point2d b, Point2d m_origin, Point2d m_dreference)
        {

            Point2d da = new Point2d(a.X - m_origin.X, a.Y - m_origin.Y);
            Point2d db = new Point2d(b.X - m_origin.X, b.Y - m_origin.Y);
            double detb = GetCross(m_dreference, db);

            // nothing is less than zero degrees
            if (detb == 0 && db.X * m_dreference.X + db.Y * m_dreference.Y >= 0) return false;

            double deta = GetCross(m_dreference, da);

            // zero degrees is less than anything else
            if (deta == 0 && da.X * m_dreference.X + da.Y * m_dreference.Y >= 0) return true;

            if (deta * detb >= 0)
            {
                // both on same side of reference, compare to each other
                return GetCross(da, db) > 0;
            }

            // vectors "less than" zero degrees are actually large, near 2 pi
            return deta > 0;
        }
        public int GetNextPoint_EdgePivoting(int prev, int current, List<int> list, double radius)
        {
            if (list.Count == 2 && prev != -1)
            {
                return list[0] + list[1] - prev;
            }
            Point2d dp;
            if (prev == -1)
                dp = new Point2d(1, 0);
            else
                //dp = Point2d.Minus(points[prev], points[current]);
                dp = new Point2d(points[current].X - points[prev].X, points[current].Y - points[prev].Y);

            int min = -1;
            for (int j = 0; j < list.Count; j++)
            {
                if (!flags[list[j]])
                {
                    if (min == -1)
                    {
                        min = list[j];
                    }
                    else
                    {
                        Point2d t = points[list[j]];
                        if (CompareAngel(points[min], t, points[current], dp) && GetDistance(t, points[current]) < radius)
                        {
                            min = list[j];
                        }
                    }
                }
            }
            //main.ShowMessage("seek P" + points[min].Index);
            return min;
        }
        public int GetNextPoint_BallPivoting(int prev, int current, List<int> list, double radius)
        {
            SortAdjListByAngel(list, prev, current);
            for (int j = 0; j < list.Count; j++)
            {
                if (flags[list[j]])
                    continue;
                int adjIndex = list[j];
                Point2d xianp = points[adjIndex];
                Point2d rightCirleCenter = GetCircleCenter(points[current], xianp, radius);
                if (!HasPointsInCircle(list, rightCirleCenter, radius, adjIndex))
                {
              //      main.DrawCircleWithXian(rightCirleCenter, points[current], points[adjIndex], radius);
                    return list[j];
                }
            }
            return -1;
        }
        private void SortAdjListByAngel(List<int> list, int prev, int current)
        {
            Point2d origin = points[current];
            Point2d df;
            if (prev != -1)
                df = new Point2d(points[prev].X - origin.X, points[prev].Y - origin.Y);
            else
                df = new Point2d(1, 0);
            int temp = 0;
            for (int i = list.Count; i > 0; i--)
            {
                for (int j = 0; j < i - 1; j++)
                {
                    if (CompareAngel(points[list[j]], points[list[j + 1]], origin, df))
                    {
                        temp = list[j];
                        list[j] = list[j + 1];
                        list[j + 1] = temp;
                    }
                }
            }
        }
        private bool HasPointsInCircle(List<int> adjPoints, Point2d center, double radius, int adjIndex)
        {
            for (int k = 0; k < adjPoints.Count; k++)
            {
                if (adjPoints[k] != adjIndex)
                {
                    int index2 = adjPoints[k];
                    if (IsInCircle(points[index2], center, radius))
                        return true;
                }
            }
            return false;
        }
        public static Point2d GetCircleCenter(Point2d a, Point2d b, double r)
        {
            double dx = b.X - a.X;
            double dy = b.Y - a.Y;
            double cx = 0.5 * (b.X + a.X);
            double cy = 0.5 * (b.Y + a.Y);
            if (r * r / (dx * dx + dy * dy) - 0.25 < 0)
            {
                return new Point2d(-1, -1);
            }
            double sqrt = Math.Sqrt(r * r / (dx * dx + dy * dy) - 0.25);
            return new Point2d((long)(cx - dy * sqrt), (long)(cy + dx * sqrt));
        }
        public static bool IsInCircle(Point2d p, Point2d center, double r)
        {
            double dis2 = (p.X - center.X) * (p.X - center.X) + (p.Y - center.Y) * (p.Y - center.Y);
            return dis2 < r * r;
        }
        public List<int>[] GetInRNeighbourList(double radius)
        {
            List<int>[] adjs = new List<int>[points.Count];
            for (int i = 0; i < points.Count; i++)
            {
                adjs[i] = new List<int>();
            }
            for (int i = 0; i < points.Count; i++)
            {

                for (int j = 0; j < points.Count; j++)
                {
                    if (i < j && distanceMap[i, j] < radius)
                    {
                        adjs[i].Add(j);
                        adjs[j].Add(i);
                    }
                }
            }
            return adjs;
        }
        private List<int> GetSortedNeighbours(int index)
        {
            List<Point2dInfo> infos = new List<Point2dInfo>(points.Count);
            for (int i = 0; i < points.Count; i++)
            {
                infos.Add(new Point2dInfo(points[i], i, distanceMap[index, i]));
            }
            infos.Sort();
            List<int> adj = new List<int>();
            for (int i = 1; i < infos.Count; i++)
            {
                adj.Add(infos[i].Index);
            }
            return adj;
        }
        public static double GetDistance(Point2d p1, Point2d p2)
        {
            return Math.Sqrt((p1.X - p2.X) * (p1.X - p2.X) + (p1.Y - p2.Y) * (p1.Y - p2.Y));
        }
        public static double GetCross(Point2d a, Point2d b)
        {
            return a.X * b.Y - a.Y * b.X;
        }
    }
}
