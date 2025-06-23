using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using Utilities.Networking.Crypting;

namespace Utilities.Networking
{
    public class UdpServer : IDisposable
    {
        public const int SIO_UDP_CONNRESET = -1744830452;

        public bool Debug { get; set; }

        public string Host
        {
            get
            {
                return this.host;
            }
            set
            {
                host = value;
            }
        }
        public int Port
        {
            get
            {
                return this.port;
            }
            set
            {
                port = value;
            }
        }
        public ICipher Crypter { get; set; }

        byte[] buffer;
        int bufferSize = 1400;
        string host = "0.0.0.0";
        int port;
        Socket serverSocket;
        Dictionary<int, UdpSession> sessions;
        System.Threading.Timer sessionClearTimer;

        public UdpServer(string host, int port, bool debug = false) : this(port, debug)
        {
            if (!String.IsNullOrEmpty(host))
                this.host = host;
        }

        public UdpServer(int port, bool debug = false)
        {
            this.Debug = debug;
            this.buffer = new byte[bufferSize];
            this.port = port;
            this.sessions = new Dictionary<int, UdpSession>();
            this.sessionClearTimer = new System.Threading.Timer(new TimerCallback(PurgeSessions), null, 60000, 60000);
        }

        void PurgeSessions(object state)
        {
            lock (sessions)
            {
                List<int> itemsToRemove = new List<int>();
                foreach (KeyValuePair<int, UdpSession> pair in sessions)
                {
                    if ((DateTime.UtcNow - pair.Value.lastData).TotalSeconds > 600)
                        itemsToRemove.Add(pair.Key);
                }

                foreach (int item in itemsToRemove)
                    sessions.Remove(item);
            }
        }

        public void Dispose()
        {
            if (serverSocket != null)
                serverSocket.Close();

            if (sessionClearTimer != null)
                sessionClearTimer.Dispose();
        }

        public void Close()
        {
            Dispose();
        }

        public void Listen()
        {
            IPEndPoint myEndpoint = new IPEndPoint(IPAddress.Parse(this.host), port);

            serverSocket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
            serverSocket.IOControl(
                (IOControlCode)SIO_UDP_CONNRESET,
                new byte[] { 0, 0, 0, 0 },
                null
            );
            serverSocket.Blocking = false;
            serverSocket.Bind(myEndpoint);

            System.Diagnostics.Debug.WriteLineIf(Debug, string.Format("UDP Server listen at {0}:{1}", IPAddress.Parse(this.host), port));

            StartReceive();
        }

        public int RegisterSession(string remoteIp, IConnectedObjectUdp obj)
        {
            UdpSession session = new UdpSession();
            session.address = IPAddress.Parse(remoteIp);
            session.obj = obj;
            obj.UdpSession = session;

            lock (sessions)
            {
                while (true)
                {
                    session.sessionId = StrongRandom.NextInt(int.MinValue, int.MaxValue);
                    if (!sessions.ContainsKey(session.sessionId))
                    {
                        sessions.Add(session.sessionId, session);
                        break;
                    }
                }
            }

            return session.sessionId;
        }

        public void UnregisterSession(int id)
        {
            lock (sessions)
                sessions.Remove(id);
        }

        void ReceiveFromCallback(IAsyncResult ar)
        {
            try
            {
                EndPoint remoteEP = new IPEndPoint(IPAddress.Any, 0);
                int bytesRead = 0;

                try
                {
                    bytesRead = serverSocket.EndReceiveFrom(ar, ref remoteEP);
                }
                catch (ObjectDisposedException)
                {
                    return;
                }
                catch (SocketException ex)
                {
                    System.Diagnostics.Debug.WriteLineIf(Debug, string.Format("Socket exception at UDPServer.ReceiveFromCallback(" + remoteEP.ToString() + "): " + ex.Message + ". " + ex.StackTrace));
                    System.Diagnostics.Debug.Assert(false, ex.Message);
                    StartReceive();
                    return;
                }

                IPEndPoint remoteEP_IP = (IPEndPoint)remoteEP;
                UdpPacket packet = UdpPacket.Parse(buffer, true, Crypter);

                StartReceive();

                UdpSession session = null;

                lock (sessions)
                {
                    if (!sessions.ContainsKey(packet.UdpSessionId))
                        throw new UdpException(string.Format("Received invalid session {2} from {0}:{1}", remoteEP_IP.Address, remoteEP_IP.Port, packet.UdpSessionId));

                    session = sessions[packet.UdpSessionId];
                }

                if (session.port > 0 && session.port != remoteEP_IP.Port)
                    throw new UdpException(string.Format("Session {0} suddenly changed the port", packet.UdpSessionId));

                session.port = remoteEP_IP.Port;

                if (packet.Sequence <= session.inSequence) //drop
                    return;

                session.inSequence = packet.Sequence;
                if (session.inSequence >= uint.MaxValue) session.inSequence = 0;

                if (session.obj == null)
                    throw new UdpException(string.Format("Session {0} object is null", packet.UdpSessionId));

                session.lastData = DateTime.UtcNow;
                session.obj.Dispatch(packet);

                System.Diagnostics.Debug.WriteLineIf(Debug, "Udp " + packet.ToString() + " received from " + remoteEP_IP.ToString());
            }
            catch (UdpException ex)
            {
                System.Diagnostics.Debug.WriteLineIf(Debug, ex.Message + "\n" + ex.StackTrace);
            }
        }

        void StartReceive()
        {
            EndPoint bindEndPoint = new IPEndPoint(IPAddress.Any, 0);
            serverSocket.BeginReceiveFrom(buffer, 0, buffer.Length, SocketFlags.None, ref bindEndPoint, new AsyncCallback(ReceiveFromCallback), null);
        }

        public void Sendpacket(UdpPacket packet, UdpSession session)
        {
            packet.Sequence = ++session.outSequence;
            byte[] sendBuffer = packet.ToByteArray(false, Crypter);
            System.Diagnostics.Debug.WriteLineIf(Debug, "Udp " + packet.ToString() + " send to " + new IPEndPoint(session.address, session.port).ToString());
            serverSocket.SendTo(sendBuffer, new IPEndPoint(session.address, session.port));
        }

    }
}
