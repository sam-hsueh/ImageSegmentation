using System;
using System.Net;
using System.Net.Sockets;
using Utilities.Networking.Crypting;

namespace Utilities.Networking
{
    public class UdpClient : IDisposable
    {

        public Socket Client
        {
            get
            {
                return socketClient;
            }
        }

        public ICipher Crypter { get; set; }

        public int UHash { get; set; }

        byte[] buffer;
        int bufferSize = 1400;
        Socket socketClient;
        EndPoint remoteEP = new IPEndPoint(IPAddress.Any, 0);

        uint in_sequence = 0;
        uint out_sequence = 0;

        public event Action<IPacket> OnIncomingPacket;


        public void Connect(string host, int port)
        {
            in_sequence = 0;
            out_sequence = 1;

            if (socketClient != null)
                socketClient.Close();

            socketClient = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
            socketClient.NoDelay = true;
            socketClient.Blocking = false;
            socketClient.ReceiveBufferSize = bufferSize;
            socketClient.SendBufferSize = bufferSize;
            socketClient.Connect(host, port);

            StartReceive();
        }

        void StartReceive()
        {
            socketClient.BeginReceiveFrom(buffer, 0, bufferSize, SocketFlags.None, ref remoteEP, new AsyncCallback(ReceiveCallback), null);
        }


        void ReceiveCallback(IAsyncResult ar)
        {
            int bytesRead = 0;
            try
            {
                bytesRead = socketClient.EndReceiveFrom(ar, ref remoteEP);
            }
            catch (ObjectDisposedException)
            {
                return;
            }
            catch (SocketException)
            {
                StartReceive();
                return;
            }

            UdpPacket p = UdpPacket.Parse(buffer, false, Crypter);
            if (p.Sequence <= in_sequence)
            {
                in_sequence = p.Sequence;
                if (OnIncomingPacket != null)
                    OnIncomingPacket(p);
            }

            StartReceive();
        }

        public void SendPacket(UdpPacket packet)
        {
            packet.Sequence = ++out_sequence;
            packet.UdpSessionId = UHash;
            byte[] data = packet.ToByteArray(true, Crypter);
            socketClient.Send(data, 0, data.Length, SocketFlags.None);
        }


        public void Close()
        {
            if (socketClient != null)
                socketClient.Close();
        }

        public void Dispose()
        {
            Close();
        }

        public UdpClient()
        {
            this.buffer = new byte[bufferSize];
        }
    }
}
