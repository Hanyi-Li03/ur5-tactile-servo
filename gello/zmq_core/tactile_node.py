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
        self._addr = f"tcp://{host}:{port}"
        self._timeout_ms = timeout_ms
        self._socket = None
        self._reset_socket()

    def _reset_socket(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.connect(self._addr)

    def read(self) -> dict[str, Any]:
        return self._request({"method": "read"})

    def read_for_recording(self, after_capture_timestamp: float | None = None,
                           target_timestamp: float | None = None) -> dict[str, Any]:
        result = self._request({
            "method": "read_for_recording",
            "after_capture_timestamp": after_capture_timestamp,
            "target_timestamp": target_timestamp,
        })
        if "latest" not in result or "recording" not in result:
            # Old nodes return a plain frame regardless of the request method.
            return {"latest": result, "recording": result}
        return result

    def _request(self, request: dict) -> dict[str, Any]:
        try:
            self._socket.send(pickle.dumps(request))
            frame = pickle.loads(self._socket.recv())
        except zmq.Again as exc:
            self._reset_socket()
            raise RuntimeError(
                f"ZMQ timeout talking to tactile server at {self._addr}. "
                "Make sure launch_nodes.py is running with tactile enabled "
                f"and responding within {self._timeout_ms} ms."
            ) from exc
        if isinstance(frame, dict) and "error" in frame:
            raise RuntimeError(frame["error"])
        return frame

    def close(self) -> None:
        self._socket.close(linger=0)
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
        try:
            self._socket.setsockopt(zmq.RCVTIMEO, 1000)
            while not self._stop_event.is_set():
                try:
                    request = pickle.loads(self._socket.recv())
                    try:
                        method = request.get("method", "read")
                        if method == "read":
                            frame = self._tactile.read()
                        elif method == "read_for_recording":
                            read = getattr(self._tactile, "read_for_recording", None)
                            if read is None:
                                latest = self._tactile.read()
                                frame = {"latest": latest, "recording": latest}
                            else:
                                target = request.get("target_timestamp")
                                frame = (read(request.get("after_capture_timestamp")) if target is None
                                         else read(request.get("after_capture_timestamp"), target_timestamp=target))
                        else:
                            raise ValueError(f"Unknown tactile method: {method}")
                    except Exception as exc:
                        frame = {"error": str(exc)}
                        print(f"Tactile read error, Tactile: {self._tactile}: {exc}")
                    self._socket.send(pickle.dumps(frame))
                except zmq.Again:
                    pass

        finally:
            self.close()

    def close(self) -> None:
        self._stop_event.set()
        self._socket.close(linger=0)
        self._context.term()

    def stop(self) -> None:
        self._stop_event.set()
