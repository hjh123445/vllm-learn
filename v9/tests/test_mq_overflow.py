"""Large media payloads must survive IPC and release after ACK or abort."""
from multiprocessing.shared_memory import SharedMemory
import pytest
from engine.mq import _channel_pair


@pytest.mark.parametrize('reverse', [False, True])
def test_large_payload_ack_release_and_slot_reuse(reverse):
    parent, child = _channel_pair(reverse=reverse, capacity=256)
    sender, receiver = (child, parent) if reverse else (parent, child)
    try:
        message = dict(type='schedule', media=bytes(range(256))*100)
        sender.send(message, wait_ack=False)
        name = sender._overflow.name
        assert receiver.recv() == message
        # Receiving alone does not release the owner's segment before ACK.
        segment = SharedMemory(name=name);segment.close()
        sender.wait_ack()
        with pytest.raises(FileNotFoundError):SharedMemory(name=name)
        for value in [dict(type='small'), message, dict(type='shutdown')]:
            sender.send(value, wait_ack=False)
            assert receiver.recv() == value
            sender.wait_ack()
        assert sender._overflow is None
    finally:
        child.close();parent.close()


def test_abort_reclaims_unacknowledged_overflow():
    sender, receiver = _channel_pair(capacity=256)
    try:
        sender.send(dict(media=b'x'*10000), wait_ack=False)
        name = sender._overflow.name
        sender.close()
        with pytest.raises(FileNotFoundError):SharedMemory(name=name)
    finally:
        receiver.close();sender.close()
