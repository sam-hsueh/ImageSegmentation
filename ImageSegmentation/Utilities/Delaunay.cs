using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace Utilities
{
    internal class Delaunay
    {
        bool Circumcircle(Point p0, Point p1, Point p2, Point center, double radius)
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

            center.X = aux1 / div;
            center.Y = aux2 / div;

            radius = Math.Sqrt((center.X - p0.X) * (center.X - p0.X) + (center.Y - p0.Y) * (center.Y - p0.Y));

            return true;
        }
    }
}
