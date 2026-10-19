from typing import Any, Dict, Optional

import numpy as np
import threading
import time
from gello.robots.robot import Robot
from gello.robots.rtde_timing import RTDETimestampTracker


class URRobot(Robot):
    """A class representing a UR robot."""

    def __init__(self, robot_ip: str = "192.168.1.10", no_gripper: bool = True):
        try:
            import rtde_control
            import rtde_receive
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing Universal Robots RTDE Python modules. Install the PyPI "
                "package with: pip install ur_rtde. The import module names are "
                "rtde_control and rtde_receive, but those are not the pip package names."
            ) from exc

        [print("in ur robot") for _ in range(4)]
        try:
            self.command_lock = threading.Lock()
            self.state_lock = threading.Lock()
            self.robot = rtde_control.RTDEControlInterface(robot_ip)
        except Exception as e:
            print(e)
            print(robot_ip)

        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        self._gripper_worker = None
        self._closed = False
        if not no_gripper:
            from gello.robots.rg2_gripper import AsyncRG2Gripper, RG2Gripper

            self.gripper = RG2Gripper(self.robot, self.command_lock)
            self._gripper_worker = AsyncRG2Gripper(self.gripper)
            self.last_gripper_pose = 0.0
            self.last_gripper_scalar = 0.0
            print("gripper connected")
            # gripper.activate()

        [print("connect") for _ in range(4)]

        self._free_drive = False
        self.robot.endFreedriveMode()
        self._use_gripper = not no_gripper
        self._timestamp_tracker = RTDETimestampTracker(self.r_inter)

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        if self._use_gripper:
            return 7
        return 6

    def _get_gripper_pos(self) -> float:
        """Return the last commanded gripper scalar from the local cache."""
        gripper_pos = self.last_gripper_scalar
        assert 0 <= gripper_pos <= 1, "Gripper position must be between 0 and 1"
        return gripper_pos

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        robot_joints = self.r_inter.getActualQ()
        if self._use_gripper:
            gripper_pos = self._get_gripper_pos()
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = robot_joints
        return pos

    def _try_control(self) -> bool:
        """Keep RPCs responsive while an RG2 script owns the control interface."""
        if self._closed:
            raise RuntimeError("UR robot is closed")
        if self._gripper_worker is not None:
            status = self._gripper_worker.status()
            if status["error"]:
                raise RuntimeError(f"RG2 error: {status['error']}")
            if status["busy"]:
                return False
        return self.command_lock.acquire(blocking=False)

    def command_joint_state(self, joint_state: np.ndarray) -> bool:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        velocity = 0.5
        acceleration = 0.5
        dt = 1.0 / 500  # 2ms
        lookahead_time = 0.2
        gain = 100

        robot_joints = joint_state[:6]
        if not self._try_control():
            return False
        try:
            t_start = self.robot.initPeriod()
            self.robot.servoJ(
                robot_joints, velocity, acceleration, dt, lookahead_time, gain
            )
            self.robot.waitPeriod(t_start)
        finally:
            self.command_lock.release()
        if self._use_gripper:
            self.command_gripper(float(joint_state[-1]))
        return True

    def get_tcp_pose(self) -> np.ndarray:
        """Return UR TCP pose as [x, y, z, rx, ry, rz]."""
        return np.asarray(self.r_inter.getActualTCPPose(), dtype=float)

    def get_tcp_force(self) -> np.ndarray:
        """Return TCP force/torque as [Fx, Fy, Fz, Tx, Ty, Tz]."""
        return np.asarray(self.r_inter.getActualTCPForce(), dtype=float)

    def inverse_kinematics(
        self,
        tcp_pose: np.ndarray,
        qnear: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return UR IK solution for TCP pose as six joint angles."""
        tcp_pose = np.asarray(tcp_pose, dtype=float).reshape(6)
        qnear_list = []
        if qnear is not None:
            qnear_list = np.asarray(qnear, dtype=float).reshape(-1)[:6].tolist()
        if not self._try_control():
            raise RuntimeError("RG2 is busy; retry inverse kinematics when it finishes")
        try:
            joints = self.robot.getInverseKinematics(tcp_pose.tolist(), qnear_list)
        finally:
            self.command_lock.release()
        joints = np.asarray(joints, dtype=float).reshape(-1)
        if joints.size != 6:
            raise RuntimeError(f"UR inverse kinematics failed for tcp_pose={tcp_pose.tolist()}")
        return joints

    def command_tcp_pose(
        self,
        tcp_pose: np.ndarray,
        velocity: float = 0.15,
        acceleration: float = 0.15,
        dt: float = 1.0 / 125,
        lookahead_time: float = 0.1,
        gain: float = 300,
    ) -> bool:
        tcp_pose = np.asarray(tcp_pose, dtype=float)
        if not self._try_control():
            return False
        try:
            t_start = self.robot.initPeriod()
            self.robot.servoL(
                tcp_pose.tolist(),
                velocity,
                acceleration,
                dt,
                lookahead_time,
                gain,
            )
            self.robot.waitPeriod(t_start)
        finally:
            self.command_lock.release()
        return True

    def command_gripper(self, gripper_scalar: float, force: float = 10.0) -> None:
        if self._closed:
            raise RuntimeError("UR robot is closed")
        if not self._use_gripper:
            return
        gripper_scalar = float(np.clip(gripper_scalar, 0.0, 1.0))
        target_width = 110.0 * (1.0 - gripper_scalar)
        if abs(gripper_scalar - self.last_gripper_scalar) > 0.03:
            self._gripper_worker.move(target_width, force)
            self.last_gripper_pose = target_width
            self.last_gripper_scalar = gripper_scalar

    def command_joint_pose(self, joint_state: np.ndarray) -> bool:
        """Command the leader robot to a given state.

        Args:
        joint_state (np.ndarray): The state to command the leader robot to.
        """
        velocity = 1
        acceleration = 1
        dt = 1.0 / 25  # 2ms
        lookahead_time = 0.2
        gain = 100

        robot_joints = joint_state[:6]
        if not self._try_control():
            return False
        try:
            self.robot.moveJ(robot_joints, velocity, acceleration)
        finally:
            self.command_lock.release()
        # self.robot.servoJ(robot_joints, velocity, acceleration, dt, lookahead_time, gain)
        # print("robot command:", robot_joints)
        if self._use_gripper:
            gripper_pos = 110 * (1 - joint_state[-1])
            if gripper_pos > 55:
                send_gripper_pos = 110
            else:
                send_gripper_pos = 0
            # send_gripper_pos = gripper_pos

            if send_gripper_pos != self.last_gripper_pose:
                self._gripper_worker.move(send_gripper_pos, 10)
                self.last_gripper_pose = send_gripper_pos
                self.last_gripper_scalar = float(np.clip(joint_state[-1], 0.0, 1.0))
            # print("gripper command: close", send_gripper_pos)
        return True

    def freedrive_enabled(self) -> bool:
        """Check if the robot is in freedrive mode.

        Returns:
            bool: True if the robot is in freedrive mode, False otherwise.
        """
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> bool:
        """Set the freedrive mode of the robot.

        Args:
            enable (bool): True to enable freedrive mode, False to disable it.
        """
        if enable and not self._free_drive:
            if not self._try_control():
                return False
            try:
                self.robot.freedriveMode()
                self._free_drive = True
            finally:
                self.command_lock.release()
        elif not enable and self._free_drive:
            if not self._try_control():
                return False
            try:
                self.robot.endFreedriveMode()
                self._free_drive = False
            finally:
                self.command_lock.release()
        return True

    def get_observations(self) -> Dict[str, Any]:
        started = time.time()
        read_timestamp = getattr(self.r_inter, "getTimestamp", None)
        native_before = read_timestamp() if read_timestamp is not None else None
        joints = self.get_joint_state()
        pos_quat = np.zeros(7)
        gripper_pos = np.array([joints[-1]])
        tcp_pose = self.get_tcp_pose()
        tcp_force = self.get_tcp_force()
        native_after = read_timestamp() if read_timestamp is not None else None
        received = time.time()
        received_monotonic = time.monotonic()
        tracker = getattr(self, "_timestamp_tracker", None)
        mapped = tracker.stamp(native_before) if tracker is not None and native_before is not None else None
        observation_timestamp = ((started+received)/2 if mapped is None
                                 else received+mapped.monotonic-received_monotonic)
        gripper_status = (
            self._gripper_worker.status() if self._gripper_worker is not None
            else {"busy": False, "error": None}
        )
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
            "tcp_force": tcp_force,
            "tcp_pose": tcp_pose,
            "gripper_busy": gripper_status["busy"],
            "gripper_error": gripper_status["error"],
            "observation_timestamp": observation_timestamp,
            "observation_received_timestamp": received,
            "rtde_timestamp": native_before,
            "rtde_timestamp_consistent": native_before == native_after and (tracker is None or mapped is not None),
            "observation_time_basis": "host_state_read" if mapped is None else "ur_rtde_clock_mapped",
            "rtde_clock_epoch": None if mapped is None else mapped.epoch,
            "rtde_clock_drift_ppm": None if mapped is None else mapped.drift_ppm,
            "rtde_observed_jitter_ms": None if mapped is None else mapped.observed_jitter_s*1000,
            "rtde_minimum_transport_delay_unknown": True,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        gripper_stopped = False
        try:
            if self._gripper_worker is not None:
                self._gripper_worker.close()
            gripper_stopped = True
        finally:
            try:
                try:
                    tracker = getattr(self, "_timestamp_tracker", None)
                    if tracker is not None:
                        tracker.close()
                finally:
                    self.r_inter.disconnect()
            finally:
                if gripper_stopped:
                    # This interface owns the RTDE script. Do not race a native
                    # custom-script call by disconnecting its control object.
                    with self.command_lock:
                        try:
                            self.robot.stopScript()
                        finally:
                            self.robot.disconnect()


def main():
    robot_ip = "192.168.1.11"
    ur = URRobot(robot_ip, no_gripper=False)
    print(ur)
    ur.set_freedrive_mode(True)
    print(ur.get_observations())


if __name__ == "__main__":
    main()
