using Utilities.Networking.Crypting;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Runtime.Serialization;
using System.Xml;

namespace Utilities.Networking
{
    public class BasePacket
    {
        public ushort Type
        {
            get { return type; }
            set { type = value; }
        }

        protected ushort type;

        protected MemoryStream stream;
        protected BinaryWriter writer;
        protected BinaryReader reader;


        public virtual byte[] GetBody()
        {
            return stream.ToArray();
        }

        protected static ushort CodePacketType(ushort type, int len)
        {
            int a1 = ((ushort)len) ^ 0xAC53;
            int a2 = ((ushort)len) ^ 0xAAAA;
            return (ushort)(type ^ a1 ^ a2);
        }


        public void WriteSerialize(object obj)
        {
            using (MemoryStream ms = new MemoryStream())
            {
                DataContractSerializer ser = new DataContractSerializer(typeof(object));
                ser.WriteObject(ms, obj);
                var data = ms.ToArray();                
                //writer.Write(data.Length);
                //writer.Write(data);
            }
        }

        public object ReadSerialized()
        {
            int len = reader.ReadInt32();
            byte[] data = reader.ReadBytes(len);
            using (MemoryStream ms = new MemoryStream(data))
            {
                using (var xmlReader = XmlDictionaryReader.CreateTextReader(ms, new XmlDictionaryReaderQuotas()))
                {
                    DataContractSerializer ser = new DataContractSerializer (typeof(object));
                    var result = ser.ReadObject(xmlReader,true);
                    return result;
                    //byte[] bytes = new byte[ms.Length];
                    //len = (int)ms.Length;
                    //ms.Read(bytes, 0, len);
                    //return bytes;
                }
            }
        }

        public byte[] ReadBytes(int count)
        {
            return reader.ReadBytes(count);
        }

        public byte ReadByte()
        {
            return reader.ReadByte();
        }

        public float ReadSingle()
        {
            return reader.ReadSingle();
        }

        public double ReadDouble()
        {
            return reader.ReadDouble();
        }

        public short ReadInt16()
        {
            return reader.ReadInt16();
        }

        public ushort ReadUInt16()
        {
            return reader.ReadUInt16();
        }

        public int ReadInt32()
        {
            return reader.ReadInt32();
        }

        public uint ReadUInt32()
        {
            return reader.ReadUInt32();
        }

        public long ReadInt64()
        {
            return reader.ReadInt64();
        }

        public ulong ReadUInt64()
        {
            return reader.ReadUInt64();
        }

        public string ReadString()
        {
            return reader.ReadString();
        }

        public void Write(byte data)
        {
            writer.Write(data);
        }

        public void Write(float data)
        {
            writer.Write(data);
        }

        public void Write(double data)
        {
            writer.Write(data);
        }

        public void Write(long data)
        {
            writer.Write(data);
        }

        public void Write(ulong data)
        {
            writer.Write(data);
        }

        public void Write(int data)
        {
            writer.Write(data);
        }

        public void Write(uint data)
        {
            writer.Write(data);
        }

        public void Write(short data)
        {
            writer.Write(data);
        }

        public void Write(ushort data)
        {
            writer.Write(data);
        }

        public void Write(string data)
        {
            writer.Write(data);
        }

        public void Write(byte[] data)
        {
            writer.Write(data);
        }

        public void Write(byte[] data, int index, int count)
        {
            writer.Write(data, index, count);
        }
	
    }
}
