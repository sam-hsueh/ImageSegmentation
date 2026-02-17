namespace Utilities.Networking.Crypting
{
    public interface ICipher
    {
        void Encrypt(ref byte[] Input, int len);
        void Decrypt(ref byte[] Input, int len);
    }
}
