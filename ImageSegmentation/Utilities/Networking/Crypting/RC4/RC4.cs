namespace Utilities.Networking.Crypting.RC4
{
    public class RC4 : ICipher
    {
        private byte[] S = new byte[256];       // The key stream state
        private uint BaseI, BaseJ;              // The indexers into the key

        // Swap the values in the given key stream
        private static void swap(byte[] S, uint x, uint y)
        {
            byte temp = S[x];
            S[x] = S[y];
            S[y] = temp;
        }

        // Get the RC4 output to be XORed with the input
        private static byte rc4_output(byte[] S, ref uint i, ref uint j)
        {
            i = (i + 1) & 255;
            j = (j + S[i]) & 255;

            swap(S, i, j);

            return S[(S[i] + S[j]) & 255];
        }

        // Initialize the keystream
        private void Init(byte[] Key, uint DropCount)
        {
            // Init to defaults: every entry gets the value of its index
            for (BaseI = 0; BaseI < 256; BaseI++)
                S[BaseI] = (byte)BaseI;

            // Swizzle the data based on the key
            for (BaseI = BaseJ = 0; BaseI < 256; BaseI++)
            {
                BaseJ = (BaseJ + Key[(int)(BaseI % Key.Length)] + S[BaseI]) & 255;
                swap(S, BaseI, BaseJ);
            }

            // Reset the indexers
            BaseI = BaseJ = 0;

            // Drop the first N bytes of the sequence
            for (uint k = 0; k < DropCount; k++)
            {
                rc4_output(S, ref BaseI, ref BaseJ);
            }
        }

        public RC4(byte[] Key, uint DropCount)
        {
            Init(Key, DropCount);
        }

        public RC4(byte[] Key)
        {
            Init(Key, 768);
        }

        // Encrypt or decrypt the byte array
        private void Crypt(ref byte[] Input, int len)
        {
            // First, create a copy of the base stream state
            byte[] s = new byte[S.Length];
            S.CopyTo(s, 0);
            uint i = BaseI, j = BaseJ;

            // XOR each entry in the data with its RC4 byte
            for (int k = 0; k < len; k++)
            {
                Input[k] = (byte)(Input[k] ^ rc4_output(s, ref i, ref j));
            }
        }

        public void Encrypt(ref byte[] input, int len)
        {
            Crypt(ref input, len);
        }

        public void Decrypt(ref byte[] input, int len)
        {
            Crypt(ref input, len);
        }

    }
}
