import pickle
import threading
from typing import Optional, Tuple

import numpy as np
import zmq

from gello.cameras.camera import CameraDriver

DEFAULT_CAMERA_PORT = 5000
DEFAULT_CLIENT_TIMEOUT_MS = 20000


class ZMQClientCamera(CameraDriver):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(
        self,
        port: int = DEFAULT_CAMERA_PORT,
        host: str = "127.0.0.1",
        timeout_ms: int = DEFAULT_CLIENT_TIMEOUT_MS,
    ):
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

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        # pack the image_size and send it to the server
        return self._request(img_size)

    def read_with_timestamp(self, img_size=None) -> dict:
        result = self._request({"method": "read_with_timestamp", "img_size": img_size})
        if not isinstance(result, dict) or "capture_timestamp" not in result:
            raise RuntimeError("Camera node has no frame times; restart launch_nodes.py")
        return result

    def read_rgb(self, img_size=None) -> np.ndarray:
        return self._request({"method": "read_rgb", "img_size": img_size})

    def read_rgb_with_timestamp(self, img_size=None) -> dict:
        result = self._request({"method": "read_rgb_with_timestamp", "img_size": img_size})
        if not isinstance(result, dict) or "capture_timestamp" not in result:
            raise RuntimeError("Camera node has no RGB frame times; restart launch_nodes.py")
        return result

    def _request(self, request):
        send_message = pickle.dumps(request)
        try:
            self._socket.send(send_message)
            state_dict = pickle.loads(self._socket.recv())
        except zmq.Again as exc:
            self._reset_socket()
            raise RuntimeError(
                f"ZMQ timeout talking to camera server at {self._addr}. "
                "Make sure launch_nodes.py is running, "
                "the camera serial number is correct, and RealSense can deliver frames "
                f"and responding within {self._timeout_ms} ms."
            ) from exc
        if isinstance(state_dict, dict) and "error" in state_dict:
            raise RuntimeError(state_dict["error"])
        return state_dict

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()


class ZMQServerCamera:
    def __init__(
        self,
        camera: CameraDriver,
        port: int = DEFAULT_CAMERA_PORT,
        host: str = "127.0.0.1",
    ):
        self._camera = camera
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        debug_message = f"Camera Sever Binding to {addr}, Camera: {camera}"
        print(debug_message)
        self._timeout_message = f"Timeout in Camera Server, Camera: {camera}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve legacy and timestamped camera requests."""
        try:
            self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
            while not self._stop_event.is_set():
                try:
                    message = self._socket.recv()
                    request = pickle.loads(message)
                    try:
                        if isinstance(request, dict):
                            method = request.get("method")
                            if method not in ("read_with_timestamp", "read_rgb_with_timestamp", "read_rgb"):
                                raise ValueError("Unknown camera request method")
                            size = request.get("img_size")
                            read = getattr(self._camera, method, None)
                            if method == "read_rgb" and read is None:
                                camera_read = self._camera.read(size)[0]
                            else:
                                read = read or getattr(self._camera, "read_with_timestamp", None)
                                if read is None:
                                    raise RuntimeError("This camera backend does not expose native frame times")
                                camera_read = read(size)
                                if method == "read_rgb_with_timestamp":
                                    camera_read = {key: value for key, value in camera_read.items() if key != "depth"}
                        else:
                            camera_read = self._camera.read(request)
                    except Exception as exc:
                        camera_read = {"error": str(exc)}
                        print(f"Camera read error, Camera: {self._camera}: {exc}")
                    self._socket.send(pickle.dumps(camera_read))
                except zmq.Again:
                    pass

        finally:
            self.close()

    def close(self) -> None:
        self._stop_event.set()
        self._socket.close(linger=0)
        self._context.term()

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()
