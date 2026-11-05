"""CPU regressions for dead-worker detection and shared-memory ACK waits."""
import multiprocessing as mp
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.worker_health import wait_for_workers
from engine.mq import _channel_pair, broadcast


def child_exit():
    raise SystemExit(7)


def consume(channel):
    assert channel.recv() == {"value": 123}
    channel.close()


class WorkerHealthTest(unittest.TestCase):
    def test_failed_worker_is_reported_without_waiting_for_deadline(self):
        child = mp.get_context("spawn").Process(target=child_exit)
        child.start()
        child.join(10)
        reader, writer = mp.Pipe(duplex=False)
        try:
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "7"):
                wait_for_workers(reader, [child], 30)
            self.assertLess(time.monotonic() - started, 1)
        finally:
            reader.close()
            writer.close()
            child.close()

    def test_pending_message_is_consumed_even_after_worker_exit(self):
        reader, writer = mp.Pipe(duplex=False)
        writer.send("ready")
        failed = type("FailedProcess", (), {"exitcode": 7})()
        try:
            wait_for_workers(reader, [failed], .1)
            self.assertEqual(reader.recv(), "ready")
        finally:
            reader.close()
            writer.close()

    def test_live_worker_without_message_times_out(self):
        reader, writer = mp.Pipe(duplex=False)
        alive = type("AliveProcess", (), {"exitcode": None})()
        try:
            with self.assertRaises(TimeoutError):
                wait_for_workers(reader, [alive], .02)
        finally:
            reader.close()
            writer.close()

    def test_broadcast_ack_with_spawned_consumer(self):
        parent, worker = _channel_pair()
        child = mp.get_context("spawn").Process(target=consume, args=(worker,))
        child.start()
        try:
            broadcast({"value": 123}, [parent],
                      wait=lambda connection: wait_for_workers(connection, [child], 10))
            child.join(10)
            self.assertEqual(child.exitcode, 0)
        finally:
            if child.is_alive():
                child.kill()
                child.join()
            parent.close()
            worker.close()
            child.close()


if __name__ == "__main__":
    unittest.main()
