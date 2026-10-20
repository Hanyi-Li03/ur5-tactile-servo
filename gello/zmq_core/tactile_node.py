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

    def _request(self, request):
        try:
            self._socket.send(pickle.dumps(request))
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

    def read(self) -> dict[str, Any]:
        return self._request({"method": "read"})

    def recording_timestamps(self, after_capture_timestamp=None) -> list[float]:
        result = self._request({"method": "recording_timestamps",
                                "after_capture_timestamp": after_capture_timestamp})
        if not isinstance(result, list):
            raise RuntimeError("Tactile node has no capture history; restart launch_nodes.py")
        return result

    def read_at(self, target_timestamp, max_skew_s=0.020, after_capture_timestamp=None):
        return self._request({"method": "read_at", "target_timestamp": target_timestamp,
                              "max_skew_s": max_skew_s,
                              "after_capture_timestamp": after_capture_timestamp})

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
                request = pickle.loads(self._socket.recv())
                try:
                    method = request.get("method", "read")
                    if method == "read":
                        frame = self._tactile.read()
                    elif method == "recording_timestamps":
                        frame = self._tactile.recording_timestamps(request.get("after_capture_timestamp"))
                    elif method == "read_at":
                        frame = self._tactile.read_at(request["target_timestamp"],
                            request.get("max_skew_s", 0.020), request.get("after_capture_timestamp"))
                    else:
                        raise ValueError(f"Unknown tactile request: {method}")
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
