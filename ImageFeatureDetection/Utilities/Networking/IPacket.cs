namespace Utilities.Networking
{
    public interface IPacket
    {
        ushort Type { get; set; }
        void WriteSerialize(object obj);
        object ReadSerialized();
        byte[] ReadBytes(int count);
        byte ReadByte();
        float ReadSingle();
        double ReadDouble();
        short ReadInt16();
        ushort ReadUInt16();
        int ReadInt32();
        uint ReadUInt32();
        long ReadInt64();
        ulong ReadUInt64();
        string ReadString();
        void Write(byte data);
        void Write(float data);
        void Write(double data);
        void Write(long data);
        void Write(ulong data);
        void Write(int data);
        void Write(uint data);
        void Write(short data);
        void Write(ushort data);
        void Write(string data);
        void Write(byte[] data);
        void Write(byte[] data, int index, int count);

        string ToString();
    }
}
