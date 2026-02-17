using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using Utilities.Networking.Crypting;

namespace Utilities.Networking
{

    public class UdpPacket : BasePacket, IPacket, IDisposable
    {
        public uint Sequence
        {
            get { return sequence; }
            set { sequence = value; }
        }

        public int UdpSessionId
        {
            get { return uhash; }
            set { uhash = value; }
        }

        uint sequence;
        int uhash;

        public UdpPacket(byte type)
        {
            base.type = type;
            base.stream = new MemoryStream();
            Init();
        }

        public UdpPacket(byte[] body, ushort type)
        {
            base.type = type;
            base.stream = new MemoryStream(body);
            Init();
        }

        void Init()
        {
            base.reader = new BinaryReader(stream);
            base.writer = new BinaryWriter(stream);
        }

        public byte[] ToByteArray(bool uhash)
        {
            return ToByteArray(uhash, null);
        }

        public byte[] ToByteArray(bool uhash, ICipher cipher)
        {
            using (MemoryStream data = new MemoryStream())
            {
                using (BinaryWriter data_writer = new BinaryWriter(data))
                {
                    byte[] body = stream.ToArray();
                    if (cipher != null)
                        cipher.Decrypt(ref body, body.Length);

                    data_writer.Write(CodePacketType(type, body.Length));
                    data_writer.Write((ushort)body.Length);
                    data_writer.Write((uint)sequence);
                    if (uhash) data_writer.Write((int)this.uhash);
                    data_writer.Write(body, 0, body.Length);

                    return data.ToArray();
                }
            }
        }

        public static UdpPacket Parse(byte[] buffer, bool uhash_e)
        {
            return Parse(buffer, uhash_e, null);
        }

        public static UdpPacket Parse(byte[] buffer, bool uhash_e, ICipher cipher)
        {
            int HEADER_SIZE = uhash_e ? 12 : 8; //WITH UHASH OR NOT
            if (buffer.Length < HEADER_SIZE) return null;
            MemoryStream instream = new MemoryStream(buffer);
            BinaryReader reader = new BinaryReader(instream);
            UdpPacket resultPacket;
            try
            {
                ushort type = reader.ReadUInt16();
                ushort len = reader.ReadUInt16();
                uint sequence = reader.ReadUInt32();
                int uhash = 0;
                if (uhash_e) uhash = reader.ReadInt32();
                if (buffer.Length - HEADER_SIZE < len) return null; //паке?не полный

                byte[] body = reader.ReadBytes((int)len);

                if (cipher != null)
                    cipher.Decrypt(ref body, (int)len);

                resultPacket = new UdpPacket(body, CodePacketType(type, len));
                resultPacket.sequence = sequence;
                resultPacket.uhash = uhash;
            }
            catch
            {
                throw;
            }
            finally
            {
                reader.Close();
                instream.Close();
            }

            return resultPacket;
        }

        public void Dispose()
        {
            if (reader != null)
            {
                reader.Close();
                //reader.Dispose();
            }
            if (writer != null)
            {
                writer.Close();
                //writer.Dispose();
            }
            if (stream != null)
            {
                stream.Close();
                stream.Dispose();
            }
        }

        public void WriteSerialize(object obj)
        {
            using (MemoryStream ms = new MemoryStream())
            {
                //BinaryFormatter bf = new BinaryFormatter();
                //bf.Serialize(ms, obj);
                //byte[] data = ms.ToArray();
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
                //BinaryFormatter bf = new BinaryFormatter();
                //return bf.Deserialize(ms);
                return null;
            }
        }

        public override string ToString()
        {
            return string.Format("UdpPacket[Type={0},Seq={1},UHash={2},Size={3}]", type, sequence, uhash, stream.Length);
        }

    }

}