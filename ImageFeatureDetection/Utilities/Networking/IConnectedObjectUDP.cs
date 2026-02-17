namespace Utilities.Networking
{
    public interface IConnectedObjectUdp
    {
        UdpSession UdpSession { get; set; }
        void Dispatch(IPacket packet);
    }
}
