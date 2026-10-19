import pickle
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import zmq

from gello.robots.robot import Robot

DEFAULT_ROBOT_PORT = 6000
DEFAULT_CLIENT_TIMEOUT_MS = 10000


class ZMQServerRobot:
    def __init__(
        self,
        robot: Robot,
        port: int = DEFAULT_ROBOT_PORT,
        host: str = "127.0.0.1",
    ):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        debug_message = f"Robot Sever Binding to {addr}, Robot: {robot}"
        print(debug_message)
        self._timout_message = f"Timeout in Robot Server, Robot: {robot}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve robot requests and always reply to rejected commands."""
        try:
            self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
            while not self._stop_event.is_set():
                try:
                    # Wait for next request from client
                    message = self._socket.recv()
                    request = pickle.loads(message)

                    # Call the appropriate method based on the request
                    method = request.get("method")
                    args = request.get("args", {})
                    result: Any
                    if method == "num_dofs":
                        result = self._robot.num_dofs()
                    elif method == "get_joint_state":
                        result = self._robot.get_joint_state()
                    elif method == "command_joint_state":
                        result = self._robot.command_joint_state(**args)
                    elif method == "get_tcp_pose":
                        if not hasattr(self._robot, "get_tcp_pose"):
                            raise NotImplementedError("Robot does not expose get_tcp_pose")
                        result = self._robot.get_tcp_pose()
                    elif method == "get_tcp_force":
                        if not hasattr(self._robot, "get_tcp_force"):
                            raise NotImplementedError("Robot does not expose get_tcp_force")
                        result = self._robot.get_tcp_force()
                    elif method == "inverse_kinematics":
                        if not hasattr(self._robot, "inverse_kinematics"):
                            raise NotImplementedError("Robot does not expose inverse_kinematics")
                        result = self._robot.inverse_kinematics(**args)
                    elif method == "command_tcp_pose":
                        if not hasattr(self._robot, "command_tcp_pose"):
                            raise NotImplementedError("Robot does not expose command_tcp_pose")
                        result = self._robot.command_tcp_pose(**args)
                    elif method == "command_gripper":
                        if not hasattr(self._robot, "command_gripper"):
                            raise NotImplementedError("Robot does not expose command_gripper")
                        result = self._robot.command_gripper(**args)
                    elif method == "get_observations":
                        result = self._robot.get_observations()
                    else:
                        result = {"error": "Invalid method"}
                        print(result)
                        raise NotImplementedError(
                            f"Invalid method: {method}, {args, result}"
                        )

                    self._socket.send(pickle.dumps(result))
                except zmq.Again:
                    # Timeout occurred - don't spam the console
                    pass
                except Exception as exc:
                    # A rejected command must reply and leave the REP loop usable.
                    self._socket.send(pickle.dumps({"error": str(exc)}))

        finally:
            self.close()

    def close(self) -> None:
        self._stop_event.set()
        self._socket.close(linger=0)
        self._context.term()

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()


class ZMQClientRobot(Robot):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(
        self,
        port: int = DEFAULT_ROBOT_PORT,
        host: str = "127.0.0.1",
        timeout_ms: int = DEFAULT_CLIENT_TIMEOUT_MS,
        blocking_gripper: bool = True,
    ):
        self._context = zmq.Context()
        self._addr = f"tcp://{host}:{port}"
        self._timeout_ms = timeout_ms
        self._blocking_gripper = blocking_gripper
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

    def _request(self, request: Dict[str, Any]) -> Any:
        try:
            self._socket.send(pickle.dumps(request))
            result = pickle.loads(self._socket.recv())
        except zmq.Again as exc:
            # Recover the next request without replaying a possibly executed command.
            self._reset_socket()
            raise RuntimeError(
                f"ZMQ timeout talking to robot server at {self._addr}. "
                "Make sure launch_nodes.py is running "
                f"and responding within {self._timeout_ms} ms."
            ) from exc
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])
        return result

    def num_dofs(self) -> int:
        """Get the number of joints in the robot.

        Returns:
            int: The number of joints in the robot.
        """
        request = {"method": "num_dofs"}
        return self._request(request)

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        request = {"method": "get_joint_state"}
        return self._request(request)

    def command_joint_state(self, joint_state: np.ndarray) -> bool | None:
        """Command the leader robot to the given state.

        Args:
            joint_state (T): The state to command the leader robot to.
        """
        request = {
            "method": "command_joint_state",
            "args": {"joint_state": joint_state},
        }
        result = self._request(request)
        if self._blocking_gripper:
            if result is False:
                raise RuntimeError("UR joint command was rejected while RG2 was busy")
            self.wait_for_gripper()
        return result

    def get_observations(self) -> Dict[str, Any]:
        """Get the current observations of the leader robot.

        Returns:
            Dict[str, np.ndarray]: The current observations of the leader robot.
        """
        request = {"method": "get_observations"}
        return self._request(request)

    def get_tcp_pose(self) -> np.ndarray:
        request = {"method": "get_tcp_pose"}
        return self._request(request)

    def get_tcp_force(self) -> np.ndarray:
        request = {"method": "get_tcp_force"}
        return self._request(request)

    def inverse_kinematics(
        self,
        tcp_pose: np.ndarray,
        qnear: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        request = {
            "method": "inverse_kinematics",
            "args": {"tcp_pose": tcp_pose, "qnear": qnear},
        }
        return self._request(request)

    def command_tcp_pose(self, tcp_pose: np.ndarray) -> bool | None:
        request = {
            "method": "command_tcp_pose",
            "args": {"tcp_pose": tcp_pose},
        }
        result = self._request(request)
        if self._blocking_gripper and result is False:
            raise RuntimeError("UR TCP command was rejected while RG2 was busy")
        return result

    def command_gripper(self, gripper_scalar: float, force: float = 10.0) -> None:
        request = {
            "method": "command_gripper",
            "args": {"gripper_scalar": gripper_scalar, "force": force},
        }
        result = self._request(request)
        if self._blocking_gripper:
            self.wait_for_gripper()
        return result

    def wait_for_gripper(self) -> None:
        """Keep synchronous inference behavior; collection opts out explicitly.

        Waiting polls status and never resends a gripper or arm command.
        """
        deadline = time.monotonic() + self._timeout_ms / 1000.0
        while True:
            status = self.get_observations()
            if status.get("gripper_error"):
                raise RuntimeError(f"RG2 error: {status['gripper_error']}")
            if not status.get("gripper_busy", False):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for RG2 completion; arm commands must stop")
            time.sleep(0.01)

    def close(self) -> None:
        """Close the ZMQ socket and context."""
        self._socket.close(linger=0)
        self._context.term()
