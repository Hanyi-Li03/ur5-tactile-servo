#!/usr/bin/env python3
"""固定点位抓取、搬运、拧螺母；不加载 checkpoint，也不运行触觉位姿模型。

TCP 使用 UR 基坐标系，单位为米和旋转向量（不是 RPY）。
抓取和接触姿态均为 [0, pi, 0]；拧紧沿用原 run_screw 的轨迹：
R = Rz(theta) @ Ry(pi)，z 按螺距下降，x/y 固定。工具轴一直竖直朝下，
但绕螺栓轴旋转时旋转向量会随 theta 改变，不能把三个旋转分量也锁死。
双 Xense 只用于超力保护和闭爪后的双侧接触检查，不做抓取偏移估计或伺服修正。

不带 --execute 仅打印固定路径，不连接硬件；实际执行示例：
    python tactile_policy/run_fixed_nutbolt_miben_lhy.py --execute
沿用原脚本默认的 4 圈、0.2 rpm、2.5 mm 螺距，拧紧耗时约 1200 秒。
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gello.force.robotiq_fts300 import AsyncForceReader
from gello.force.robotiq_fts300_stream import RobotiqFTS300Stream
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.tactile_node import ZMQClientTactile


# 两个标定位置固定在代码中，不接受模型补偿或命令行覆盖。
CALIBRATED_TCP_POSE_BASE = np.asarray(
    [-0.0635, -0.5499, 0.1372, 0.0, np.pi, 0.0], dtype=np.float64
)
ABS_TCP_BD = [0.0365, -0.5474, 0.142, 0.0, np.pi, 0.0]
BOLT_TCP_CONTACT_POSE = np.asarray(ABS_TCP_BD, dtype=np.float64)
BOLT_AXIS_BASE = np.asarray([0.0, 0.0, 1.0])
CONTACT_WAIT_S = 1.0
TACTILE_FORCE_SAFETY_LIMIT_N = np.asarray([25.0, 25.0, 25.0])
TACTILE_TORQUE_SAFETY_LIMIT_NMM = np.asarray([100.0, 100.0, 100.0])
FORCE_TARE_SECONDS = 2.0


class SafetyStop(RuntimeError):
    """数据无效时停止发送运动命令。"""


class ForceLimitStop(SafetyStop):
    """触觉或六维力超过原有硬限值。"""


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=6001)
    parser.add_argument("--robot-timeout-ms", type=int, default=10000)
    parser.add_argument("--left-tactile-port", type=int, default=5100)
    parser.add_argument("--right-tactile-port", type=int, default=5101)
    parser.add_argument("--tactile-timeout-ms", type=int, default=15000)
    parser.add_argument("--approach-height-mm", type=float, default=50.0)
    parser.add_argument("--transport-height-mm", type=float, default=80.0)
    parser.add_argument("--retreat-height-mm", type=float, default=50.0)
    parser.add_argument("--gripper-force", type=float, default=4.0)
    parser.add_argument("--gripper-settle-s", type=float, default=1.0)
    parser.add_argument("--move-duration-s", type=float, default=6.0)
    parser.add_argument("--contact-duration-s", type=float, default=6.0)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--turns", type=float, default=4.0)
    parser.add_argument("--screw-rpm", type=float, default=0.2)
    parser.add_argument("--screw-pitch-mm", type=float, default=2.5)
    parser.add_argument("--rotation-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--screw-print-every", type=int, default=20)
    parser.add_argument("--max-command-position-step-mm", type=float, default=0.5)
    parser.add_argument("--max-command-rotation-step-deg", type=float, default=0.5)
    parser.add_argument(
        "--min-tactile-peak", type=float, default=1e-4,
        help="沿用原 full-cycle 的双侧接触判据；闭爪后两侧六维触觉范数均须达到此值。",
    )
    parser.add_argument(
        "--use-force-safety", "--use-ur-force-safety", dest="use_force_safety",
        action=argparse.BooleanOptionalAction, default=False,
        help="额外启用 FT300-S 连续流或 UR RTDE 六维力保护；双 Xense 保护始终启用。",
    )
    parser.add_argument(
        "--force-source", choices=("fts300-stream", "ur-rtde"), default="fts300-stream"
    )
    parser.add_argument("--force-serial-port", default=None)
    parser.add_argument("--force-serial-timeout-s", type=float, default=0.1)
    parser.add_argument("--force-max-age-s", type=float, default=0.1)
    parser.add_argument(
        "--force-serial-low-latency", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--force-hard-stop-n", type=float, default=18.0)
    parser.add_argument("--torque-hard-stop-nm", type=float, default=2.0)
    parser.add_argument(
        "--open-gripper-on-force-stop", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--execute", action="store_true", help="连接硬件并执行固定动作。")
    parser.add_argument("--yes", action="store_true", help="跳过原脚本的 RUN 确认。")
    args = parser.parse_args(argv)
    for name in (
        "robot_timeout_ms", "tactile_timeout_ms", "approach_height_mm",
        "transport_height_mm", "retreat_height_mm", "gripper_force",
        "move_duration_s", "contact_duration_s", "control_hz", "turns",
        "screw_rpm", "screw_pitch_mm", "screw_print_every",
        "max_command_position_step_mm", "max_command_rotation_step_deg",
        "min_tactile_peak",
        "force_serial_timeout_s", "force_max_age_s", "force_hard_stop_n",
        "torque_hard_stop_nm",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if not np.isfinite(args.gripper_settle_s) or args.gripper_settle_s < 0:
        parser.error("--gripper-settle-s must be nonnegative and finite")
    # 一次采样不可跨越半圈，否则仅凭 TCP 姿态无法区分目标转向/圈数。
    angle_step = args.screw_rpm * 2.0 * np.pi / (60.0 * args.control_hz)
    descent_step_mm = args.screw_rpm * args.screw_pitch_mm / (60.0 * args.control_hz)
    if angle_step >= np.pi or angle_step > np.deg2rad(args.max_command_rotation_step_deg):
        parser.error("screw rpm/control hz exceeds the configured rotation step limit")
    if descent_step_mm > args.max_command_position_step_mm:
        parser.error("screw rpm/pitch/control hz exceeds the configured position step limit")
    return args


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = rotvec / angle
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    antisymmetric = np.asarray([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    sine = float(np.linalg.norm(antisymmetric) / 2.0)
    angle = float(np.arctan2(sine, cosine))
    if angle < 1e-12:
        return np.zeros(3)
    if np.pi - angle < 1e-5:
        values, vectors = np.linalg.eigh((rotation + rotation.T) / 2.0)
        axis = vectors[:, int(np.argmax(values))]
        if np.dot(axis, antisymmetric) < 0:
            axis = -axis
        return axis * angle
    return angle / (2.0 * sine) * antisymmetric


def interpolate_pose(start: np.ndarray, target: np.ndarray, blend: float) -> np.ndarray:
    start_R = rotvec_to_matrix(start[3:])
    relative = matrix_to_rotvec(start_R.T @ rotvec_to_matrix(target[3:]))
    rotation = start_R @ rotvec_to_matrix(blend * relative)
    return np.r_[start[:3] + blend * (target[:3] - start[:3]), matrix_to_rotvec(rotation)]


def limit_pose_step(args, current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    dp = desired[:3] - current[:3]
    current_R = rotvec_to_matrix(current[3:])
    drot = matrix_to_rotvec(rotvec_to_matrix(desired[3:]) @ current_R.T)
    position_limit = args.max_command_position_step_mm / 1000.0
    rotation_limit = np.deg2rad(args.max_command_rotation_step_deg)
    if np.linalg.norm(dp) <= position_limit and np.linalg.norm(drot) <= rotation_limit:
        return desired.copy()
    dp *= min(1.0, position_limit / max(np.linalg.norm(dp), 1e-12))
    drot *= min(1.0, rotation_limit / max(np.linalg.norm(drot), 1e-12))
    return np.r_[current[:3] + dp, matrix_to_rotvec(rotvec_to_matrix(drot) @ current_R)]


def checked_pose(robot) -> np.ndarray:
    pose = np.asarray(robot.get_tcp_pose(), dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise SafetyStop("robot returned an invalid TCP pose")
    return pose


class SafetyMonitor:
    """只读取力保护数据，不读取模型输入，不调用推理。"""

    def __init__(self, args, robot, resources: ExitStack) -> None:
        self.args = args
        self.robot = robot
        self.force_bias = np.zeros(6)
        self.tactile = []
        for port in (args.left_tactile_port, args.right_tactile_port):
            source = ZMQClientTactile(
                port=port, host=args.host, timeout_ms=args.tactile_timeout_ms
            )
            resources.callback(source.close)
            self.tactile.append(source)
        self.force_reader = None
        if args.use_force_safety and args.force_source == "fts300-stream":
            stream = RobotiqFTS300Stream(
                port=args.force_serial_port, timeout_s=args.force_serial_timeout_s,
                low_latency=args.force_serial_low_latency,
            )
            try:
                self.force_reader = AsyncForceReader(stream, max_age_s=args.force_max_age_s)
            except BaseException:
                stream.close()
                raise
            resources.callback(self.force_reader.close)
            self.force_reader.wait_until_ready()

    @staticmethod
    def valid_wrench(value, source: str) -> np.ndarray:
        wrench = np.asarray(value, dtype=np.float64)
        if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
            raise SafetyStop(f"{source} returned an invalid six-axis wrench")
        return wrench

    def read_wrench(self) -> np.ndarray:
        value = (self.force_reader.read() if self.force_reader is not None
                 else self.robot.get_tcp_force())
        return self.valid_wrench(value, self.args.force_source)

    def tare(self) -> None:
        if not self.args.use_force_safety:
            return
        print(f"Taring {self.args.force_source} for {FORCE_TARE_SECONDS:g}s...")
        samples = []
        count = max(10, int(round(FORCE_TARE_SECONDS * self.args.control_hz)))
        for _ in range(count):
            samples.append(self.read_wrench())
            time.sleep(FORCE_TARE_SECONDS / count)
        self.force_bias = np.mean(samples, axis=0)

    def check(self) -> tuple[dict, dict]:
        frames = tuple(source.read() for source in self.tactile)
        for name, frame in zip(("left", "right"), frames):
            if not frame.get("valid", False):
                raise SafetyStop(f"{name} Xense frame is invalid")
            wrench = self.valid_wrench(frame.get("force6d"), f"{name} Xense")
            if np.any(np.abs(wrench[:3]) >= TACTILE_FORCE_SAFETY_LIMIT_N):
                raise ForceLimitStop(f"{name} tactile force limit exceeded: {wrench[:3]}")
            if np.any(np.abs(wrench[3:]) >= TACTILE_TORQUE_SAFETY_LIMIT_NMM):
                raise ForceLimitStop(f"{name} tactile torque limit exceeded: {wrench[3:]}")
        if self.args.use_force_safety:
            wrench = self.read_wrench() - self.force_bias
            if np.any(np.abs(wrench[:3]) >= self.args.force_hard_stop_n):
                raise ForceLimitStop(f"{self.args.force_source} force limit exceeded: {wrench[:3]}")
            if np.any(np.abs(wrench[3:]) >= self.args.torque_hard_stop_nm):
                raise ForceLimitStop(f"{self.args.force_source} torque limit exceeded: {wrench[3:]}")
        return frames

    def require_grasp_contact(self) -> None:
        """保留原 predict_valid_offset 的接触门槛，但不调用位姿模型。"""
        frames = self.check()
        peaks = tuple(float(np.linalg.norm(frame["force6d"])) for frame in frames)
        if min(peaks) < self.args.min_tactile_peak:
            raise SafetyStop(f"insufficient bilateral tactile contact after grasp: peaks={peaks}")
        print(f"Bilateral grasp contact confirmed: peaks={peaks}", flush=True)


def wait_checked(args, safety, duration: float) -> None:
    """静止等待期间继续检查力；到位等待至少 duration 秒。"""
    deadline = time.monotonic() + duration
    while True:
        safety.check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(1.0 / args.control_hz, remaining))


def command_pose(args, robot, pose: np.ndarray) -> None:
    if args.execute:
        robot.command_tcp_pose(pose)


def move_pose(args, robot, safety, target: np.ndarray, duration: float) -> None:
    start = checked_pose(robot)
    steps = max(1, int(np.ceil(duration * args.control_hz)))
    for index in range(1, steps + 1):
        tick = time.monotonic()
        safety.check()
        phase = index / steps
        blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
        command_pose(args, robot, interpolate_pose(start, target, blend))
        time.sleep(max(0.0, 1.0 / args.control_hz - (time.monotonic() - tick)))


def fixed_waypoints(args) -> list[tuple[str, np.ndarray, float]]:
    grasp = CALIBRATED_TCP_POSE_BASE.copy()
    approach = grasp.copy()
    approach[2] += args.approach_height_mm / 1000.0
    lift = grasp.copy()
    lift[2] += args.transport_height_mm / 1000.0
    contact = BOLT_TCP_CONTACT_POSE.copy()
    above = contact.copy()
    above[2] += args.approach_height_mm / 1000.0
    transfer = above.copy()
    transfer[2] = max(lift[2], above[2])
    return [
        ("grasp approach", approach, args.move_duration_s),
        ("fixed grasp", grasp, args.contact_duration_s),
        ("lift", lift, args.move_duration_s),
        ("transfer", transfer, args.move_duration_s),
        ("bolt approach", above, args.move_duration_s),
        ("fixed bolt contact", contact, args.contact_duration_s),
    ]


def full_cycle_grasp(args, robot, safety) -> None:
    safety.check()
    if args.execute:
        robot.command_gripper(0.0, force=args.gripper_force)
    wait_checked(args, safety, args.gripper_settle_s)
    safety.tare()
    for index, (name, target, duration) in enumerate(fixed_waypoints(args)):
        print(f"{name}: {np.round(target, 6)}", flush=True)
        move_pose(args, robot, safety, target, duration)
        if index == 1:
            safety.check()
            if args.execute:
                robot.command_gripper(1.0, force=args.gripper_force)
            wait_checked(args, safety, args.gripper_settle_s)
            safety.require_grasp_contact()
    print(f"At ABS_TCP_BD; waiting {CONTACT_WAIT_S:g}s before run_screw.", flush=True)
    wait_checked(args, safety, CONTACT_WAIT_S)


def screw_pose(args, phase: float) -> np.ndarray:
    """原 run_screw 的名义轨迹，所有抓取偏移和伺服补偿均为零。"""
    theta = args.rotation_sign * args.turns * 2.0 * np.pi * phase
    pose = BOLT_TCP_CONTACT_POSE.copy()
    pose[:3] -= args.turns * args.screw_pitch_mm / 1000.0 * phase * BOLT_AXIS_BASE
    # Rz(theta) @ Ry(pi) 的连续旋转向量，避免 pi 附近从旋转矩阵提取轴的跳号。
    pose[3:] = [-np.pi * np.sin(theta / 2.0), np.pi * np.cos(theta / 2.0), 0.0]
    return pose


def run_screw(args, robot, safety) -> None:
    duration = args.turns * 60.0 / args.screw_rpm
    steps = max(1, int(np.ceil(duration * args.control_hz)))
    for index in range(steps + 1):
        tick = time.monotonic()
        safety.check()
        phase = index / steps
        desired = screw_pose(args, phase)
        command = limit_pose_step(args, checked_pose(robot), desired)
        command_pose(args, robot, command)
        if index % args.screw_print_every == 0 or index == steps:
            print(
                f"screw {index}/{steps} turns={args.turns * phase:.4f} "
                f"theta={args.rotation_sign * args.turns * 360.0 * phase:+.2f}deg "
                f"advance={args.turns * args.screw_pitch_mm * phase:.3f}mm "
                f"target_z={desired[2]:.6f}m", flush=True,
            )
        time.sleep(max(0.0, 1.0 / args.control_hz - (time.monotonic() - tick)))


def release_and_retreat(args, robot, safety) -> None:
    safety.check()
    if args.execute:
        robot.command_gripper(0.0, force=args.gripper_force)
    wait_checked(args, safety, args.gripper_settle_s)
    retreat = checked_pose(robot)
    retreat[2] += args.retreat_height_mm / 1000.0
    move_pose(args, robot, safety, retreat, args.move_duration_s)


def main() -> None:
    args = parse_args()
    for name, target, _ in fixed_waypoints(args):
        print(f"Plan {name}: {np.round(target, 6)}")
    print(
        f"Wait at contact: {CONTACT_WAIT_S:g}s; screw: {args.turns:g} turns, "
        f"{args.screw_rpm:g}rpm, {args.turns * args.screw_pitch_mm:g}mm descent, "
        f"{args.turns * 60.0 / args.screw_rpm:g}s; then release and retreat."
    )
    if not args.execute:
        print("Plan only; no hardware connection. Add --execute to run.")
        return
    with ExitStack() as resources:
        robot = ZMQClientRobot(
            port=args.robot_port, host=args.host, timeout_ms=args.robot_timeout_ms
        )
        resources.callback(robot.close)
        try:
            if robot.num_dofs() != 7:
                raise SafetyStop("expected a 7-DoF UR5 + RG2 robot server")
            safety = SafetyMonitor(args, robot, resources)
            safety.tare()
            safety.check()
            print("Current TCP [m,rotvec]:", np.round(checked_pose(robot), 6))
            if not args.yes and input("Real UR5 motion will start. Type RUN: ").strip() != "RUN":
                print("Cancelled without sending a motion command.")
                return
            full_cycle_grasp(args, robot, safety)
            run_screw(args, robot, safety)
            release_and_retreat(args, robot, safety)
            print("Fixed-pose screw sequence completed.")
        except ForceLimitStop as exc:
            print(f"FORCE LIMIT STOP: {exc}. No further arm commands will be sent.")
            if args.open_gripper_on_force_stop:
                try:
                    robot.command_gripper(0.0, force=args.gripper_force)
                except Exception as open_error:
                    print(f"Failed to open gripper after force stop: {open_error}")
            raise
        except KeyboardInterrupt:
            print("Interrupted; no further motion commands will be sent.")
        except (SafetyStop, RuntimeError) as exc:
            print(f"STOP: {exc}. No further motion commands will be sent.")
            raise


if __name__ == "__main__":
    main()
