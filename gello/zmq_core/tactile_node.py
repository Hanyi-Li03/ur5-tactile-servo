import pickle
import threading
from typing import Any

import zmq


DEFAULT_TACTILE_PORT = 5100
DEFAULT_CLIENT_TIMEOUT_MS = 15000


class ZMQClientTactile:
    def __init__(
        self,
        port: int = DEFAULT_TACTILE_PORT,
        host: str = "127.0.0.1",
        timeout_ms: int = DEFAULT_CLIENT_TIMEOUT_MS,
    ) -> None:
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._addr = f"tcp://{host}:{port}"
        self._timeout_ms = timeout_ms
        self._socket.connect(self._addr)

    def read(self) -> dict[str, Any]:
        try:
            self._socket.send(pickle.dumps({"method": "read"}))
            frame = pickle.loads(self._socket.recv())
        except zmq.Again as exc:
            raise RuntimeError(
                f"ZMQ timeout talking to tactile server at {self._addr}. "
                "Make sure launch_nodes.py is running with tactile enabled "
                f"and responding within {self._timeout_ms} ms."
            ) from exc
        if isinstance(frame, dict) and "error" in frame:
            raise RuntimeError(frame["error"])
        return frame

    def close(self) -> None:
        self._socket.close()
        self._context.term()


class ZMQServerTactile:
    def __init__(
        self,
        tactile: Any,
        port: int = DEFAULT_TACTILE_PORT,
        host: str = "127.0.0.1",
    ) -> None:
        self._tactile = tactile
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        print(f"Tactile Server Binding to {addr}, Tactile: {tactile}")
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)
        while not self._stop_event.is_set():
            try:
                self._socket.recv()
                try:
                    frame = self._tactile.read()
                except Exception as exc:
                    frame = {"error": str(exc)}
                    print(f"Tactile read error, Tactile: {self._tactile}: {exc}")
                self._socket.send(pickle.dumps(frame))
            except zmq.Again:
                pass

    def stop(self) -> None:
        self._stop_event.set()
        close = getattr(self._tactile, "close", None)
        if callable(close):
            close()
