"""One SDK process per sensor, with a single overwriteable shared frame."""

import multiprocessing as mp
import pickle
import time

import numpy as np

from gello.tactile.xense import (
    XENSE_DEFAULT_FORCE_HEIGHT,
    XENSE_DEFAULT_FORCE_WIDTH,
    XenseTactileReader,
)


_FRAME_CAPACITY = 8 * 1024 * 1024


def _publish(result, buffer, size, lock, available):
    payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > len(buffer):
        raise RuntimeError(f"Xense frame exceeds shared buffer capacity ({len(payload)} bytes)")
    with lock:
        np.frombuffer(buffer, dtype=np.uint8)[:len(payload)] = np.frombuffer(payload, dtype=np.uint8)
        size.value = len(payload)
        available.set()


def _xense_worker(reader_kwargs, reader_factory, buffer, size, lock, available,
                  stop, startup, poll_interval_s):
    reader = None
    ready = False
    try:
        if reader_factory is None:
            # OpenCV's own pool competes with the bounded ONNX pools. This
            # setting belongs only to the tactile child, not RGB or policy code.
            import cv2

            cv2.setNumThreads(1)
            reader_factory = XenseTactileReader
        reader = reader_factory(**reader_kwargs)
        startup.send(None)
        startup.close()
        ready = True
        while not stop.is_set():
            try:
                _publish({"frame": reader.read()}, buffer, size, lock, available)
            except Exception as exc:
                _publish({"error": str(exc)}, buffer, size, lock, available)
                if stop.wait(0.02):
                    break
            if poll_interval_s > 0:
                stop.wait(poll_interval_s)
    except KeyboardInterrupt:
        pass
    except BaseException as exc:
        try:
            if not ready:
                startup.send(str(exc))
            else:
                _publish({"error": str(exc)}, buffer, size, lock, available)
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        startup.close()
        if reader is not None:
            reader.close()


class ProcessXenseTactileReader:
    """Keep SDK state, inference and device release in an isolated process.

    ``read`` is used by AsyncXenseTactileReader's cache worker. Clients still
    read that cache immediately. Shared memory holds only the newest result;
    a slow parent never accumulates a queue of old tactile images.
    """

    def __init__(
        self,
        serial_number: str | None = None,
        *,
        cam_id: int | None = None,
        force_height: int = XENSE_DEFAULT_FORCE_HEIGHT,
        force_width: int = XENSE_DEFAULT_FORCE_WIDTH,
        name: str = "xense",
        poll_interval_s: float = 0.0,
        startup_timeout_s: float = 20.0,
        _reader_factory=None,
    ) -> None:
        if not np.isfinite(poll_interval_s) or poll_interval_s < 0:
            raise ValueError("Xense poll_interval_s must be finite and nonnegative")
        if not np.isfinite(startup_timeout_s) or startup_timeout_s <= 0:
            raise ValueError("Xense startup_timeout_s must be finite and positive")
        self._name = name
        self._closed = False
        # Never fork a process which already owns RTDE, camera or ZMQ threads.
        context = mp.get_context("spawn")
        self._buffer = context.RawArray("B", _FRAME_CAPACITY)
        self._size = context.RawValue("Q", 0)
        self._lock = context.Lock()
        self._available = context.Event()
        self._stop = context.Event()
        parent_startup, child_startup = context.Pipe(duplex=False)
        self._process = context.Process(
            target=_xense_worker,
            args=(dict(serial_number=serial_number, cam_id=cam_id,
                       force_height=force_height, force_width=force_width, name=name),
                  _reader_factory, self._buffer, self._size, self._lock,
                  self._available, self._stop, child_startup, poll_interval_s),
            name=f"xense-{name}",
            daemon=True,
        )
        try:
            self._process.start()
            child_startup.close()
            if not parent_startup.poll(startup_timeout_s):
                raise RuntimeError(f"{name} SDK process startup timed out")
            try:
                error = parent_startup.recv()
            except EOFError as exc:
                raise RuntimeError(f"{name} SDK process exited during startup") from exc
            if error is not None:
                raise RuntimeError(f"{name} SDK process startup failed: {error}")
        except BaseException:
            self.close()
            raise
        finally:
            parent_startup.close()
            child_startup.close()
        print(f"Xense {name}: process backend ready (pid={self._process.pid})")

    def read(self) -> dict:
        if self._closed:
            raise RuntimeError(f"{self._name} process reader is closed")
        deadline = time.monotonic() + 0.5
        while True:
            if self._closed:
                raise RuntimeError(f"{self._name} process reader is closed")
            if not self._process.is_alive():
                raise RuntimeError(
                    f"{self._name} SDK process exited (code={self._process.exitcode}); restart the node"
                )
            if self._available.wait(0.05):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{self._name} SDK process is not producing frames")
        # A native crash must not strand the cache thread on a process lock.
        if not self._lock.acquire(timeout=0.1):
            raise RuntimeError(f"{self._name} SDK shared frame is unavailable")
        try:
            payload = bytes(memoryview(self._buffer).cast("B")[:self._size.value])
            self._available.clear()
        finally:
            self._lock.release()
        result = pickle.loads(payload)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["frame"]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._process.pid is None:
            return
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=1.0)
        if self._process.is_alive():
            raise RuntimeError(f"{self._name} SDK process did not stop")
