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
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._addr = f"tcp://{host}:{port}"
        self._timeout_ms = timeout_ms
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
        send_message = pickle.dumps(img_size)
        try:
            self._socket.send(send_message)
            state_dict = pickle.loads(self._socket.recv())
        except zmq.Again as exc:
            raise RuntimeError(
                f"ZMQ timeout talking to camera server at {self._addr}. "
                "Make sure launch_nodes.py is running, "
                "the camera serial number is correct, and RealSense can deliver frames "
                f"and responding within {self._timeout_ms} ms."
            ) from exc
        if isinstance(state_dict, dict) and "error" in state_dict:
            raise RuntimeError(state_dict["error"])
        return state_dict


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
        """Serve the leader robot state over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv()
                img_size = pickle.loads(message)
                try:
                    camera_read = self._camera.read(img_size)
                except Exception as exc:
                    camera_read = {"error": str(exc)}
                    print(f"Camera read error, Camera: {self._camera}: {exc}")
                self._socket.send(pickle.dumps(camera_read))
            except zmq.Again:
                pass

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()
