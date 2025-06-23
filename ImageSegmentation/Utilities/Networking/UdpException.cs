using System;

namespace Utilities.Networking
{
    class UdpException : Exception
    {
        public UdpException()
        {
        }

        public UdpException(string message)
            : base(message)
        {
        }
    }
}
