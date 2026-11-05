"""Bound control-plane waits by worker health and a configurable timeout."""
import time


def wait_for_workers(connection, processes, timeout, stop_event=None):
    deadline = time.monotonic() + timeout
    while not connection.poll(0):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Worker IPC wait cancelled")
        failed = [(rank, p.exitcode) for rank, p in enumerate(processes)
                  if p.exitcode is not None]
        if failed:
            raise RuntimeError(f"GPU worker exited while waiting for IPC: {failed}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"GPU worker IPC exceeded {timeout:g}s")
        connection.poll(min(.1, remaining))
