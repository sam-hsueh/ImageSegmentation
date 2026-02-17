using System;
using System.Security.Cryptography;

namespace Utilities.Networking
{
    public class StrongRandom
    {
        public static int NextInt(int min, int max)
        {
            using (var rng = RandomNumberGenerator.Create())
            {
                byte[] buffer = new byte[4];
                rng.GetBytes(buffer);
                int result = BitConverter.ToInt32(buffer, 0);
                return new Random(result).Next(min, max);
            }
        }
    }
}
