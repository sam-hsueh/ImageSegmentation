namespace Utilities.Networking
{
    public interface IConnectedObject
    {
        void Dispatch(IPacket packet);
        void ConnectionDropped();
    }
}
