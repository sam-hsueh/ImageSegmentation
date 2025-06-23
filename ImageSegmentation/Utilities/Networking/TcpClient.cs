using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Utilities.Networking
{
    public class TcpClient : IDisposable
    {
        //public delegate void DOnError(TcpClient tcpClient, SocketException ex);
        //public delegate void DOnIncomingPacket(float[] x, float[] y);
        //public delegate void DOnConnect(TcpClient tcpClient);
        //public delegate void DOnDisconnect(TcpClient tcpClient);

        public enum TcpClientStatus
        {
            Disconnected,
            Connecting,
            Connected
        }

        const int defaultBufferSize = ushort.MaxValue * 2;

        public bool Connected
        {
            get
            {
                lock (tcpClient)
                {
                    if (tcpClient == null || tcpClient.Client == null)
                        return false;
                    return tcpClient.Connected;
                }
            }
        }

        public TcpClientStatus Status { get; private set; }

        public Socket Socket
        {
            get
            {
                return tcpClient.Client;
            }
        }

        public int ReceiveBufferSize
        {
            get
            {
                lock (tcpClient)
                    return tcpClient.ReceiveBufferSize;
            }
            set
            {
                if (tcpClient == null || tcpClient.Client == null)
                    return;
                lock (tcpClient)
                    tcpClient.ReceiveBufferSize = value;
            }
        }
        //public int SendBufferSize
        //{
        //    get
        //    {
        //        lock (tcpClient)
        //            return tcpClient.SendBufferSize;
        //    }
        //    set
        //    {
        //        lock (tcpClient)
        //            tcpClient.SendBufferSize = value;
        //    }
        //}
        public event Action<TcpClient, SocketException> OnError;
        public event Action<float[], float[]> OnIncomingPacket;
        public event Action<TcpClient> OnConnect;
        public event Action<TcpClient> OnDisconnect;

        public System.Net.Sockets.TcpClient tcpClient;
        byte[] buffer;
        byte[] tempBuffer;
        bool blocking;
        public float[] X, Z;

        AutoResetEvent connectEvent;

        public void ConnectAsync(string host, int port)
        {
            ReInitSocket();
            tcpClient.BeginConnect(host, port, new AsyncCallback(ConnectCallback), null);
            Status = TcpClientStatus.Connecting;
            SendData(OpenBuffer);
//            SendData(CloseBuffer);
        }

        public void Connect(string host, int port)
        {
            if (tcpClient == null || tcpClient.Client == null)
                return;
            ConnectAsync(host, port);
            connectEvent.Reset();
            connectEvent.WaitOne();
        }

        public void Close()
        {
            Status = TcpClientStatus.Disconnected;

            if (tcpClient == null || tcpClient.Client == null || !tcpClient.Connected)
                return;

            OnDiconnectInternal();
        }

        void OnErrorInternal(SocketException ex)
        {
            if (tcpClient == null || tcpClient.Client == null)
                return;
            Status = TcpClientStatus.Disconnected;

            lock (tcpClient)
            {
                tcpClient.Close();
            }
            if (ex.ErrorCode == 10054)
                OnDiconnectInternal();
            if (OnError != null)
                OnError(this, ex);
        }

        void OnDiconnectInternal()
        {
            if (tcpClient == null || tcpClient.Client == null)
                return;
            Status = TcpClientStatus.Disconnected;
            lock (tcpClient)
            {
                tcpClient.Close();
                if (OnDisconnect != null)
                    OnDisconnect(this);
            }
        }

        void ConnectCallback(IAsyncResult ar)
        {
            if (tcpClient == null || tcpClient.Client == null)
                return;
            try
            {
                lock (tcpClient)
                    tcpClient.EndConnect(ar);

                BeginReceive();

                Status = TcpClientStatus.Connected;
                connectEvent.Set();

                SendData(GetDataBuffer);
                if (OnConnect != null)
                    OnConnect(this);
            }
            catch (SocketException ex)
            {
                OnErrorInternal(ex);
            }
        }

        void BeginReceive()
        {
            lock (tcpClient)
            {
                if (tcpClient.Client == null)
                    return;
                tcpClient.Client.BeginReceive(tempBuffer, 0, tempBuffer.Length, SocketFlags.None, ReceiveCallback, null);
            }
        }

        unsafe void ReceiveCallback(IAsyncResult ar)
        {
            if (tcpClient == null || tcpClient.Client == null)
                return;
            try
            {
                int bytesRead = 0;
                lock (tcpClient)
                {

                    bytesRead = tcpClient.Client.EndReceive(ar);
                    if (bytesRead == 0)
                        OnDiconnectInternal();
                }
                lock (buffer)
                {
                    byte[] newbuffer = new byte[buffer.Length + bytesRead];
                    buffer.CopyTo(newbuffer, 0);
                    Array.Copy(tempBuffer, 0, newbuffer, buffer.Length, bytesRead);
                    buffer = newbuffer;
                    Parse(ref buffer);
                }
                BeginReceive();
            }
            catch (SocketException ex)
            {
                OnErrorInternal(ex);
            }
        }
        public unsafe void Parse(ref byte[] buffer)
        {
            if (buffer.Length < 4007) return;
            else if (buffer.Length == 8014)
            {
                var nbuffer = buffer[^4007..];
                buffer = nbuffer;
            }
            int k = buffer.Length;
            var newbuffer = new byte[k];
            buffer.CopyTo(newbuffer, 0);
            //Array.Copy(buffer, nbuffer, k);
            //MemoryStream instream = new MemoryStream(buffer);
            //BinaryReader reader = new BinaryReader(instream);
            try
            {
                var h = newbuffer[4] << 8;
                int ps = h + newbuffer[5];
                if (BCCChecked(newbuffer))
                {
                    X = new float[ps];
                    Z = new float[ps];
                    fixed (byte* src = newbuffer)
                    fixed (float* x = X)
                    fixed (float* z = Z)
                    {
                        var ptr = src + 6;
                        var cx = x;
                        var cz = z;
                        for (int i = 0; i < ps; i++, ptr += 4, cx++, cz++)
                        {
                            short sx = 0, sy = 0;
                            //sx = (short)(ptr[1] << 8 | ptr[0] & 0xff);
                            //sy = (short)(ptr[3] << 8 | ptr[2] & 0xff);
                            sx = (short)((ptr[0] & 0xff) | (ptr[1] & 0xff) << 8);
                            sy = (short)((ptr[2] & 0xff) | (ptr[3] & 0xff) << 8);
                            *cx = (sx / 100F);
                            *cz = (sy / 100F);
                        }
                    }
                    if (OnIncomingPacket != null)
                        OnIncomingPacket.Invoke(X, Z);
                }
                try
                {
                if (k > 4007)
                    buffer = buffer[4007..k];
                else if (k == 4007)
                    buffer = new byte[0];
                }
                catch { }
            }
            catch
            {
                throw;
            }
            finally
            {
                //reader.Close();
                //instream.Close();
            }
        }
        public bool BCCChecked(byte[] data)
        {
            byte CheckCode = 0;
            int n = data.Length;
            for (int i = 0; i < n - 1; i++)
            {
                CheckCode ^= data[i];
            }
            return CheckCode == data[n - 1];
        }
        public byte[] GetDataBuffer = new byte[] { 0xF0, 0x5A, 0x00, 0x45, 0xEF };
        public byte[] OpenBuffer = new byte[] { 0xF0, 0x5A, 0x00, 0x11, 0x01, 0xBA };
        public byte[] CloseBuffer = new byte[] { 0xF0, 0x5A, 0x00, 0x11, 0x02, 0xB9 };

        public void SendData(byte[] sendBuffer)
        {
            if (tcpClient == null || tcpClient.Client == null)
                return;
            try
            {
                if (!Connected) return;

                lock (tcpClient)
                {
                    buffer = new byte[0];
                    tcpClient.Client.BeginSend(sendBuffer, 0, sendBuffer.Length, SocketFlags.None, null, null);
                }
            }
            catch (ObjectDisposedException)
            { }
        }

        public void Dispose()
        {
            try
            {

                if (tcpClient != null)
                {
                    lock (tcpClient)
                    {

                        tcpClient.Close();
                        tcpClient.Dispose();
                    }
                }


                if (connectEvent != null)
                    lock (connectEvent)
                    {
                        connectEvent.Close();
                        connectEvent = null;
                    }
            }
            catch { }
        }

        void ReInitSocket()
        {
            Dispose();

            Status = TcpClientStatus.Disconnected;
            connectEvent = new AutoResetEvent(false);
            tcpClient = new System.Net.Sockets.TcpClient();
            tcpClient.Client.Blocking = blocking;
            tcpClient.Client.NoDelay = true;
            buffer = new byte[0];
            tempBuffer = new byte[ReceiveBufferSize];
        }

        public TcpClient(bool blocking = false)
        {
            this.blocking = blocking;
            this.ReInitSocket();
            this.ReceiveBufferSize = defaultBufferSize;
            //this.SendBufferSize = defaultBufferSize;
        }
    }
}
