using System;
using System.Net.Sockets;
using System.Threading;
using System.Net;
using Utilities.Networking.Crypting;

namespace Utilities.Networking
{
	public class TcpConnection : IDisposable
	{
        public int ConnectionId { get; set; }

        public IConnectedObject ConnectedObject
        {
            get
            {
                return obj;
            }
            set
            {
                obj = value;
            }
        }

        public Socket Socket
        {
            get
            {
                return socket;
            }
        }

        ICipher cipher;
        Socket socket;
        byte[] recv_buffer;
		byte[] storage_buffer;
		TcpServer parent;
        IConnectedObject obj;
        public bool closed
        {
            set;
            get;
        }
		
		public TcpConnection(TcpServer parent, Socket socket, int bufferSize, Type cipherType)
		{
            if (cipherType != null)
                cipher = Activator.CreateInstance(cipherType) as ICipher;
			this.socket = socket;
			this.recv_buffer = new byte[bufferSize];
			this.storage_buffer = new byte[0];
			this.parent = parent;
            this.ConnectionId = StrongRandom.NextInt(1, int.MaxValue);
			
            socket.BeginReceive(recv_buffer, 0,
                    recv_buffer.Length, SocketFlags.None,
                    new AsyncCallback(ReceiveCallback), null);
		}
		
  		public void Close()
        {
            if (closed) return;
            closed = true;

            if (obj != null)
                obj.ConnectionDropped();
           
            lock (parent.connections) 
                parent.connections.Remove(this);

            System.Diagnostics.Debug.WriteLine(string.Format("{0} disconnected. Connections left: {1}", 
                this.obj == null ? "Unknown" : this.obj.ToString(), parent.connections.Count));

            if (socket != null)
            {
                socket.Close();
                socket = null;
            }
            
            obj = null;
        }

        public void Dispose()
        {
            Close();
        }

        public override string ToString()
        {
            if (socket != null && socket.RemoteEndPoint != null)
                return string.Format("TcpConnection[address={0}, connected={1}]", socket.RemoteEndPoint.ToString(), socket.Connected.ToString());
            else
                return string.Format("TcpConnection[Closed]");
        }

        public bool IsValidUDPSession(UdpSession session)
        {
            if (session.address.ToString() != ((IPEndPoint)socket.RemoteEndPoint).Address.ToString()) return false;
            if (session.sessionId != ConnectionId) return false;
            if (obj == null) return false;

            return true;
        }
				
		internal void ReceiveCallback(IAsyncResult result)
        {
            try
            {
                int bytesRead = 0;
                
                if (socket != null)
                    bytesRead = socket.EndReceive(result);

                if (bytesRead == 0)
				{
					Close();
					return;
				}

                //byte[] newbuffer = new byte[storage_buffer.Length + bytesRead];
                //            storage_buffer.CopyTo(newbuffer, 0);
                //            Array.Copy(recv_buffer, 0, newbuffer, storage_buffer.Length, bytesRead);
                //            storage_buffer = newbuffer;
                byte[] newbuffer = new byte[bytesRead];
                Array.Copy(recv_buffer, 0, newbuffer, 0, bytesRead);
                storage_buffer = newbuffer;

                if (storage_buffer.Length == 128 || storage_buffer.Length == 64)
                {
                    parent.ReceiveData = (byte[])storage_buffer.Clone();
                    //parent.OnNotAuthorizedPacketInternal(this, (byte[])storage_buffer.Clone());
                }
                
                //TcpPacket packet = null;
                //while ((packet = TcpPacket.Parse(ref storage_buffer, cipher)) != null)
                //{

                //    if (obj == null)
                //        parent.OnNotAuthorizedPacketInternal(this, packet);
                //    else
                //        obj.Dispatch(packet);
                //}

                if (socket != null)
                    socket.BeginReceive(recv_buffer, 0,
                         recv_buffer.Length, SocketFlags.None,
                         new AsyncCallback(ReceiveCallback), null);

            }
			catch (ObjectDisposedException) { 
                Close();
            }
			catch (SocketException) { 
                Close();
            }
        }
		
		
		public void Send(TcpPacket packet)
		{
            if (socket == null) 
                return;

            try
            {
                byte[] sendBuffer = packet.ToByteArray(cipher);
                //socket.BeginSend(sendBuffer, 0, sendBuffer.Length, SocketFlags.None, new AsyncCallback(SendCallback), packet);
                socket.Send(sendBuffer);
            }
            catch (ObjectDisposedException)
            { }
            catch (SocketException)
            {
                 Close();
                 return;
            }
		}
        public void Send(byte[] packet)
        {
            if (socket == null)
                return;

            try
            {
                byte[] sendBuffer = packet;
                //socket.BeginSend(sendBuffer, 0, sendBuffer.Length, SocketFlags.None, new AsyncCallback(SendCallback), packet);
                socket.Send(sendBuffer);
            }
            catch (ObjectDisposedException)
            { }
            catch (SocketException)
            {
                Close();
                return;
            }
        }


        void SendCallback(IAsyncResult ar)
        {
            try
            {
                if (socket == null) return;
                socket.EndSend(ar);
            }
            catch (ObjectDisposedException) { }
            catch (SocketException)
            {
                Close();
                return;
            }
        }
	}
}

