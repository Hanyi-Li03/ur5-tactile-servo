#!/usr/bin/env python3
"""Run tactile-servo nut screwing on a real UR5 + RG2 + dual Xense sensors.

Supply an external trained checkpoint with ``--checkpoint``. Collection uses
``real_ur5rg2/collect_lerobot_xenstactile_servo_control.py``.
Despite the historical dataset key name, its label is

    T_reference_tcp^-1 @ T_current_tcp

represented as local xyz + RPY.  Therefore a zero prediction is the standard
nut grasp.  It is not the MuJoCo ``T_nut_tip`` label.

For safety this program never sends a motion command unless ``--execute`` is
provided.  The default ``screw-only`` mode assumes the nut is already grasped
and just touching the bolt.  ``full-cycle`` additionally uses explicitly
configured grasp and bolt-contact TCP poses.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import select
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.tactile_node import ZMQClientTactile
from gello.force.robotiq_fts300 import (
    AsyncForceReader,
    RobotiqFTS300ModbusSerial,
    RobotiqFTS300ModbusTCP,
)
from gello.force.robotiq_fts300_stream import RobotiqFTS300Stream
from tactile_policy.data import Normalization, signed_log1p
from tactile_policy.model import create_model


XENSE_LEFT_SERIAL = "OG001783"
XENSE_RIGHT_SERIAL = "OG001782"
XENSE_FORCE_HEIGHT = 35
XENSE_FORCE_WIDTH = 20
TACTILE_FORCE_SAFETY_LIMIT_N = np.asarray([25.0, 25.0, 25.0])
TACTILE_TORQUE_SAFETY_LIMIT_NMM = np.asarray([100.0, 100.0, 100.0])
FORCE_TARE_SECONDS = 2.0
CALIBRATED_TCP_POSE_BASE = np.asarray(
    [ 0.107776706, -0.696389763,  0.199836926, 0.0, np.pi, 0.0],
    dtype=np.float64,
)

ABS_TCP_BD = [0.0375347175, -0.5471017194, 0.144, 0.0, np.pi, 0.0]
BOLT_TCP_CONTACT_POSE = np.asarray(
    ABS_TCP_BD,
    dtype=np.float64,
)

class SafetyStop(RuntimeError):
    """Raised when a real-hardware safety condition is violated."""

class ForceLimitStop(SafetyStop):
    """Raised only when a tactile or configured wrench limit is exceeded."""

def six_floats(value: str) -> np.ndarray:
    values = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if values.shape != (6,):
        raise argparse.ArgumentTypeError("expected six comma-separated values")
    return values


def three_floats(value: str) -> np.ndarray:
    values = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if values.shape != (3,):
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to an external trained checkpoint (not bundled with this project).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=6001)
    parser.add_argument("--robot-timeout-ms", type=int, default=10000)
    parser.add_argument("--left-tactile-port", type=int, default=5100)
    parser.add_argument("--right-tactile-port", type=int, default=5101)
    parser.add_argument("--tactile-timeout-ms", type=int, default=15000)
    parser.add_argument("--xense-force-height", type=int, default=XENSE_FORCE_HEIGHT)
    parser.add_argument("--xense-force-width", type=int, default=XENSE_FORCE_WIDTH)
    parser.add_argument("--xense-left-serial", default=XENSE_LEFT_SERIAL)
    parser.add_argument("--xense-right-serial", default=XENSE_RIGHT_SERIAL)

    parser.add_argument(
        "--mode", choices=("screw-only", "full-cycle", "validate-pose"), default="screw-only"
    )
    parser.add_argument("--validation-rounds", type=int, default=10)
    parser.add_argument(
        "--validation-reference-tcp", type=six_floats,
        default=six_floats("0.0375347175,-0.5471017194,0.137,0,3.141592653589793,0"),
        help="Collection label reference TCP [m,rotvec]; used only for ground truth.",
    )
    parser.add_argument(
        "--validation-reset-reference-tcp", type=six_floats,
        default=six_floats("0.0375347175,-0.5471017194,0.137,0,3.141592653589793,0"),
        help="Collection CALIBRATED_TCP_POSE_BASE anchoring the random reset geometry.",
    )
    parser.add_argument("--validation-seed", type=int, default=None)
    parser.add_argument("--validation-close-wait-s", type=float, default=0.5)
    parser.add_argument("--validation-timeout-s", type=float, default=30.0)
    parser.add_argument("--validation-max-joint-step-rad", type=float, default=0.3)
    parser.add_argument(
        "--grasp-placement-test",
        action="store_true",
        help=(
            "Run only the real-nut grasp-offset placement test: move to the "
            "grasp pose, close RG2, wait for a manual nut perturbation, infer "
            "the local TCP offset, and apply it to the standard TCP pose."
        ),
    )
    parser.add_argument(
        "--placement-test-prediction-samples",
        type=int,
        default=100,
        help=(
            "Deprecated compatibility option. Placement correction now runs "
            "as a real-time tactile servo."
        ),
    )
    parser.add_argument(
        "--placement-test-print-every",
        type=int,
        default=1,
        help="Print placement-servo telemetry every N control steps.",
    )
    parser.add_argument(
        "--placement-test-servo-duration-s",
        type=float,
        default=0.0,
        help=(
            "Placement tactile-servo duration in seconds. The default 0 runs "
            "continuously until Ctrl+C; use a positive value for a timed test."
        ),
    )
    parser.add_argument(
        "--placement-test-max-position-mm",
        type=float,
        default=5.0,
        help="Abort the placement test if predicted translation exceeds this norm.",
    )
    parser.add_argument(
        "--placement-test-max-rotation-deg",
        type=float,
        default=10.0,
        help="Abort the placement test if predicted rotation exceeds this angle.",
    )
    parser.add_argument(
        "--grasp-tcp-pose",
        type=six_floats,
        default=CALIBRATED_TCP_POSE_BASE.copy(),
        help="Standard grasp TCP [x,y,z,rx,ry,rz] in the UR base frame.",
    )
    parser.add_argument(
        "--bolt-contact-tcp-pose",
        type=six_floats,
        default=BOLT_TCP_CONTACT_POSE.copy(),
        help=(
            "Standard-grasp TCP pose when the nut bottom just touches the bolt. "
            "Required in full-cycle mode."
        ),
    )
    parser.add_argument("--approach-height-mm", type=float, default=50.0)
    parser.add_argument("--transport-height-mm", type=float, default=80.0)
    parser.add_argument("--retreat-height-mm", type=float, default=50.0)
    parser.add_argument("--gripper-force", type=float, default=4.0)
    parser.add_argument("--gripper-settle-s", type=float, default=1.0)
    parser.add_argument("--move-duration-s", type=float, default=6.0)
    parser.add_argument("--contact-duration-s", type=float, default=6.0)

    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument(
        "--screw-print-every",
        type=int,
        default=1,
        help="Print screw telemetry every N control steps.",
    )
    parser.add_argument("--turns", type=float, default=4.0)
    parser.add_argument("--screw-rpm", type=float, default=0.2)
    parser.add_argument("--screw-pitch-mm", type=float, default=2.5)
    parser.add_argument("--rotation-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--bolt-axis-base", type=three_floats, default=three_floats("0,0,1"))
    parser.add_argument(
        "--servo-start-turns",
        type=float,
        default=0.0,
        help="Enable tactile feedback only after this many screw turns.",
    )
    parser.add_argument("--servo-ref", type=six_floats, default=six_floats("0,0,0,0,0,0"))
    parser.add_argument("--servo-kp", type=six_floats, default=six_floats("1.0,1.0,1.0,0.2,0.2,0"))
    parser.add_argument("--servo-filter-alpha", type=float, default=0.05)
    parser.add_argument("--max-servo-position-mm", type=float, default=2.0)
    parser.add_argument("--max-servo-rotation-deg", type=float, default=2.0)
    parser.add_argument("--max-command-position-step-mm", type=float, default=0.5)
    parser.add_argument("--max-command-rotation-step-deg", type=float, default=0.5)
    parser.add_argument("--min-tactile-peak", type=float, default=1e-4)

    parser.add_argument(
        "--use-force-safety",
        "--use-ur-force-safety",
        dest="use_force_safety",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hard-stop on excessive wrench from --force-source.",
    )
    parser.add_argument(
        "--force-source",
        choices=("fts300-stream", "modbus-serial", "modbus-tcp", "ur-rtde"),
        default="fts300-stream",
        help="Read FT300-S 100 Hz stream (default), Modbus RTU/TCP, or UR RTDE.",
    )
    parser.add_argument("--force-serial-port", default=None)
    parser.add_argument("--force-serial-slave-address", type=int, default=9)
    parser.add_argument("--force-serial-register-address", type=int, default=180)
    parser.add_argument("--force-serial-function-code", type=int, default=3)
    parser.add_argument("--force-serial-baudrate", type=int, default=19200)
    parser.add_argument(
        "--force-serial-timeout-s",
        type=float,
        default=0.1,
        help="Serial response timeout; samples are received in the background.",
    )
    parser.add_argument("--force-serial-retries", type=int, default=5)
    parser.add_argument(
        "--force-serial-low-latency", action=argparse.BooleanOptionalAction,
        default=True, help="Temporarily enable adapter low-latency mode for FT streaming.",
    )
    parser.add_argument(
        "--force-max-age-s", type=float, default=0.1,
        help="Maximum background force sample age; stale/error samples stop inference.",
    )
    parser.add_argument(
        "--force-serial-wakeup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force-modbus-host", default="192.168.1.1")
    parser.add_argument("--force-modbus-port", type=int, default=502)
    parser.add_argument("--force-modbus-unit-id", type=int, default=9)
    parser.add_argument("--force-modbus-address", type=int, default=0)
    parser.add_argument(
        "--force-modbus-function", choices=("holding", "input"), default="input"
    )
    parser.add_argument(
        "--force-modbus-format",
        choices=("int16", "int32", "float32"),
        default="int32",
    )
    parser.add_argument(
        "--force-modbus-word-order", choices=("big", "little"), default="big"
    )
    parser.add_argument("--force-modbus-scale", type=float, default=1.0)
    parser.add_argument("--force-modbus-timeout-s", type=float, default=1.0)
    parser.add_argument("--force-hard-stop-n", type=float, default=18.0)
    # 力安全保护
    parser.add_argument(
        "--open-gripper-on-force-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the gripper when a tactile or wrench hard limit is exceeded.",
    )
    parser.add_argument("--torque-hard-stop-nm", type=float, default=2.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the final RUN prompt.")
    args = parser.parse_args()
    for name in ("force_max_age_s", "force_serial_timeout_s"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")

    if args.mode == "validate-pose":
        if args.grasp_placement_test:
            parser.error("validate-pose cannot be combined with --grasp-placement-test")
        for name in ("validation_rounds", "validation_timeout_s", "validation_max_joint_step_rad"):
            if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                parser.error(f"--{name.replace('_', '-')} must be positive and finite")
        if not np.isfinite(args.validation_close_wait_s) or args.validation_close_wait_s < 0:
            parser.error("--validation-close-wait-s must be nonnegative and finite")
        if not np.all(np.isfinite(args.validation_reset_reference_tcp)):
            parser.error("--validation-reset-reference-tcp must be finite")
        if args.validation_seed is not None and not 0 <= args.validation_seed < 2**32:
            parser.error("--validation-seed must be in [0, 2**32)")
        if not np.all(np.isfinite(args.validation_reference_tcp)):
            parser.error("--validation-reference-tcp must be finite")

    if args.mode == "full-cycle" and args.bolt_contact_tcp_pose is None:
        parser.error("--bolt-contact-tcp-pose is required with --mode full-cycle")
    for name in (
        "control_hz", "turns", "screw_rpm", "screw_pitch_mm", "move_duration_s",
        "contact_duration_s", "max_command_position_step_mm",
        "max_command_rotation_step_deg", "screw_print_every",
        "placement_test_prediction_samples", "placement_test_print_every",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.servo_filter_alpha <= 1.0:
        parser.error("--servo-filter-alpha must be in (0,1]")
    if args.placement_test_max_position_mm <= 0.0:
        parser.error("--placement-test-max-position-mm must be positive")
    if args.placement_test_max_rotation_deg <= 0.0:
        parser.error("--placement-test-max-rotation-deg must be positive")
    if args.placement_test_servo_duration_s < 0.0:
        parser.error("--placement-test-servo-duration-s must be nonnegative")
    axis_norm = float(np.linalg.norm(args.bolt_axis_base))
    if axis_norm < 1e-9:
        parser.error("--bolt-axis-base must be nonzero")
    args.bolt_axis_base = args.bolt_axis_base / axis_norm
    return args


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(
        path.expanduser().resolve(), map_location=device, weights_only=False
    )
    model_type = checkpoint.get("model_type", "simple_cnn")
    # 新增：仅深度 checkpoint 进入新分支，其他模型保留原 create_model 路径。
    if model_type == "native_depth_cnn":
        # 延迟导入：仅使用分布力时不要求已有深度网络。
        from tactile_policy.model import NativeDepthCNN

        depth_norm = Normalization.from_dict(checkpoint["normalization"])
        preprocessing = checkpoint.get("input_preprocessing", {})
        expected_keys = ["observation.tactile_left.depth", "observation.tactile_right.depth"]
        if (checkpoint.get("input_channels") != 2
                or checkpoint.get("input_keys") != expected_keys
                or preprocessing.get("channel_order") != ["left", "right"]
                or preprocessing.get("png_channel") != 0
                or preprocessing.get("resize") is not None
                or preprocessing.get("signed_log1p") is not False
                or preprocessing.get("per_image_normalization") is not False):
            raise ValueError("Expected left/right single-channel depth without network resizing")
        _validate_live_depth_scale(preprocessing, depth_norm)
        input_hw = tuple(checkpoint["model_input_hw"])
        model = NativeDepthCNN(in_channels=2, out_dim=6, input_hw=input_hw).to(device)
        # 新增：供 predict_offset 选择输入分支，不改变已有调用和返回接口。
        model.tactile_input_kind = "depth"
    else:
        model = create_model(
            model_type,
            grid_size=int(checkpoint.get("grid_size", 32)),
            image_size=int(checkpoint.get("model_image_size", 128)),
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    norm = Normalization.from_dict(checkpoint["normalization"])
    print(
        f"Loaded {model_type} epoch={checkpoint.get('epoch')} "
        f"target_key={checkpoint.get('target_key')} device={device}"
    )
    return model, norm


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    axis_hat = skew(rotvec / angle)
    return (
        np.eye(3)
        + np.sin(angle) * axis_hat
        + (1.0 - np.cos(angle)) * (axis_hat @ axis_hat)
    )


def matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return np.zeros(3)
    if np.pi - angle < 1e-5:
        # Eigenvector with eigenvalue one remains stable near pi.
        values, vectors = np.linalg.eig(rotation)
        axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return axis * angle
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * np.sin(angle))
    return axis * angle


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    return matrix_to_rotvec(np.asarray(rotation, dtype=np.float64))


def _fit_force(value, height: int, width: int) -> tuple[np.ndarray, bool]:
    output = np.zeros((height, width, 3), dtype=np.float32)
    if value is None:
        return output, False
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return output, False
    if array.ndim == 3 and array.shape[-1] == 3:
        h, w = min(height, array.shape[0]), min(width, array.shape[1])
        output[:h, :w] = array[:h, :w, :3]
        return output, True
    try:
        flattened = array.reshape(-1, 3)
    except ValueError:
        return output, False
    count = min(len(flattened), height * width)
    output.reshape(-1, 3)[:count] = flattened[:count]
    return output, count > 0


def format_tactile_frame(frame: dict, height: int, width: int) -> dict:
    force, force_ok = _fit_force(frame.get("force"), height, width)
    force6d = np.asarray(
        frame.get("force6d", np.zeros(6)), dtype=np.float32
    ).reshape(-1)
    formatted_wrench = np.zeros(6, dtype=np.float32)
    formatted_wrench[: min(6, len(force6d))] = force6d[:6]
    return {
        "force": force,
        "force6d": formatted_wrench,
        # 新增：保留 SDK 原始深度，仅深度模型会执行后续图像转换。
        "depth": frame.get("depth"),
        "valid": bool(frame.get("valid", force_ok)) and force_ok,
    }


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    return rotvec_to_matrix(np.asarray(axis, dtype=np.float64) * float(angle))


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    rx = axis_rotation(np.array([1.0, 0.0, 0.0]), roll)
    ry = axis_rotation(np.array([0.0, 1.0, 0.0]), pitch)
    rz = axis_rotation(np.array([0.0, 0.0, 1.0]), yaw)
    return rz @ ry @ rx


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to RPY using ``Rz(yaw) Ry(pitch) Rx(roll)``."""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    pitch = np.arctan2(
        -rotation[2, 0],
        np.hypot(rotation[2, 1], rotation[2, 2]),
    )
    return np.asarray(
        [
            np.arctan2(rotation[2, 1], rotation[2, 2]),
            pitch,
            np.arctan2(rotation[1, 0], rotation[0, 0]),
        ],
        dtype=np.float64,
    )


def smoothstep5(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def clipped(vector: np.ndarray, maximum: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= maximum or magnitude < 1e-12:
        return vector
    return vector * (maximum / magnitude)


def interpolate_pose(start: np.ndarray, target: np.ndarray, blend: float) -> np.ndarray:
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    start_R = rotvec_to_matrix(start[3:])
    target_R = rotvec_to_matrix(target[3:])
    relative = start_R.T @ target_R
    rotation = start_R @ rotvec_to_matrix(blend * rotation_vector(relative))
    return np.r_[start[:3] + blend * (target[:3] - start[:3]), matrix_to_rotvec(rotation)]


def limit_pose_step(
    current: np.ndarray,
    target: np.ndarray,
    max_position_step: float,
    max_rotation_step: float,
) -> np.ndarray:
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    dp = clipped(target[:3] - current[:3], max_position_step)
    current_R = rotvec_to_matrix(current[3:])
    target_R = rotvec_to_matrix(target[3:])
    drot = clipped(rotation_vector(target_R @ current_R.T), max_rotation_step)
    return np.r_[current[:3] + dp, matrix_to_rotvec(rotvec_to_matrix(drot) @ current_R)]


class TactileSource:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.left = ZMQClientTactile(
            port=args.left_tactile_port,
            host=args.host,
            timeout_ms=args.tactile_timeout_ms,
        )
        self.right = ZMQClientTactile(
            port=args.right_tactile_port,
            host=args.host,
            timeout_ms=args.tactile_timeout_ms,
        )

    def close(self) -> None:
        self.left.close()
        self.right.close()

    def read(self) -> tuple[dict, dict]:
        # Take both snapshots before converting either tactile image.
        frames = (self.left.read(), self.right.read())
        values = []
        for frame in frames:
            values.append(
                format_tactile_frame(
                    frame,
                    self.args.xense_force_height,
                    self.args.xense_force_width,
                )
            )
        if not all(bool(item["valid"]) for item in values):
            raise SafetyStop("one or both Xense tactile frames are invalid")
        return values[0], values[1]


def format_force_torque(force_torque: np.ndarray) -> np.ndarray:
    array = np.asarray(force_torque, dtype=np.float32).reshape(-1)
    if array.size < 6:
        array = np.pad(array, (0, 6 - array.size))
    return array[:6].astype(np.float32)


class ForceReader:
    """Read current wrench values without serial transactions in the inference loop."""

    def __init__(self, args: argparse.Namespace, robot: ZMQClientRobot) -> None:
        self.source = args.force_source
        self._robot = robot
        self._modbus = None
        self._background = None
        self._closed = False
        if self.source == "fts300-stream":
            self._modbus = RobotiqFTS300Stream(
                port=args.force_serial_port,
                slave_address=args.force_serial_slave_address,
                baudrate=args.force_serial_baudrate,
                timeout_s=args.force_serial_timeout_s,
                low_latency=args.force_serial_low_latency,
            )
        elif self.source == "modbus-serial":
            self._modbus = RobotiqFTS300ModbusSerial(
                port=args.force_serial_port,
                slave_address=args.force_serial_slave_address,
                register_address=args.force_serial_register_address,
                function_code=args.force_serial_function_code,
                baudrate=args.force_serial_baudrate,
                timeout_s=args.force_serial_timeout_s,
                wakeup=args.force_serial_wakeup,
                retries=args.force_serial_retries,
            )
        elif self.source == "modbus-tcp":
            self._modbus = RobotiqFTS300ModbusTCP(
                host=args.force_modbus_host,
                port=args.force_modbus_port,
                unit_id=args.force_modbus_unit_id,
                register_address=args.force_modbus_address,
                function=args.force_modbus_function,
                value_format=args.force_modbus_format,
                word_order=args.force_modbus_word_order,
                scale=args.force_modbus_scale,
                timeout_s=args.force_modbus_timeout_s,
            )

        if self.source in ("fts300-stream", "modbus-serial"):
            try:
                self._background = AsyncForceReader(self._modbus, max_age_s=args.force_max_age_s)
                self._background.wait_until_ready()
            except BaseException:
                self.close()
                raise

    def read(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Force reader is closed")
        if self._background is not None:
            return self._background.read()
        if self.source in ("modbus-serial", "modbus-tcp"):
            if self._modbus is None:
                raise RuntimeError("Modbus force reader is not initialized")
            return format_force_torque(self._modbus.read())
        if self.source == "ur-rtde":
            return format_force_torque(self._robot.get_tcp_force())
        raise RuntimeError(f"Unsupported force source: {self.source}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._background is not None:
            # The worker cancels pending reads, joins, and closes its own port.
            self._background.close()
        else:
            close = getattr(self._modbus, "close", None)
            if close is not None:
                close()


# 新增：独立保留采集脚本的灰度转换，迁移时无需携带采集脚本。
# SDK 深度 -> 采集灰度 uint8 -> 左右两个通道；无需保存 PNG。
def _resize_float_map(array: np.ndarray, height: int, width: int) -> np.ndarray:
    if array.shape == (height, width):
        return array.astype(np.float32, copy=False)
    # 新增：仅深度图尺寸不同时需要 Pillow。
    from PIL import Image
    image = Image.fromarray(array.astype(np.float32, copy=False))
    image = image.resize((width, height))
    return np.asarray(image, dtype=np.float32)

def _fit_xense_depth_array(
    value: np.ndarray | None,
    depth_height: int,
    depth_width: int,
) -> tuple[np.ndarray, bool]:
    formatted = np.zeros((depth_height, depth_width, 3), dtype=np.uint8)
    if value is None:
        return formatted, False

    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return formatted, False

    if array.size == 0:
        return formatted, False

    array = np.squeeze(array)
    if array.ndim == 3:
        if array.shape[-1] in (1, 3, 4):
            array = array[..., 0]
        elif array.shape[0] in (1, 3, 4):
            array = array[0]
    if array.ndim == 2:
        try:
            array = _resize_float_map(array, depth_height, depth_width)
        except (TypeError, ValueError):
            height = min(array.shape[0], depth_height)
            width = min(array.shape[1], depth_width)
            resized = np.zeros((depth_height, depth_width), dtype=np.float32)
            resized[:height, :width] = np.asarray(
                array[:height, :width],
                dtype=np.float32,
            )
            array = resized
        image = _to_uint8_image(array)
        return np.repeat(image[..., None], 3, axis=-1), True

    flattened = np.asarray(array, dtype=np.float32).reshape(-1)
    count = min(flattened.shape[0], depth_height * depth_width)
    map_2d = np.zeros((depth_height, depth_width), dtype=np.float32)
    map_2d.reshape(-1)[:count] = flattened[:count]
    image = _to_uint8_image(map_2d)
    return np.repeat(image[..., None], 3, axis=-1), count > 0

def _to_uint8_image(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array

    array = np.asarray(array, dtype=np.float32)
    finite = array[np.isfinite(array)]
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    if finite.size > 0 and float(finite.max()) <= 1.0:
        array = np.clip(array, 0.0, 1.0) * 255.0
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def _validate_live_depth_scale(preprocessing: dict, norm: Normalization) -> None:
    """新增：与离线深度预测相同，只接受已支持的固定尺度 checkpoint。"""
    mode = preprocessing.get("type")
    if mode not in ("depth_png_uint8_div255", "depth_png_uint8_train_max"):
        raise ValueError(f"Unsupported depth preprocessing: {mode!r}")
    divisor = float(preprocessing.get("divisor", float("nan")))
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("Depth divisor must be finite and positive")
    if mode == "depth_png_uint8_div255" and divisor != 255.0:
        raise ValueError("div255 preprocessing requires divisor=255")
    if (norm.input_mean.shape != (2,) or norm.input_std.shape != (2,)
            or not np.all(norm.input_mean == 0)
            or not np.isfinite(norm.input_std).all()
            or not np.all(norm.input_std > 0)
            or not np.allclose(norm.input_std, divisor, rtol=1e-6, atol=0)):
        raise ValueError("Depth checkpoint normalization disagrees with saved divisor")


def _live_depth_channel(value, input_hw) -> np.ndarray:
    """新增：缺失/无效深度明确报错，避免将空图输入实时控制模型。"""
    if value is None:
        raise SafetyStop("Depth model requires SDK frame['depth'] on both sensors")
    array = np.asarray(value)
    if array.size == 0 or not np.isfinite(array).all():
        raise SafetyStop("Empty or nonfinite tactile depth frame")
    image, valid = _fit_xense_depth_array(array, *input_hw)
    if not valid:
        raise SafetyStop("Invalid tactile depth frame")
    return image[..., 0]


def predict_offset(model, norm: Normalization, left: dict, right: dict, device):
    # 新增：深度走采集同款灰度转换；原分布力处理完整保留在 else 中。
    if getattr(model, "tactile_input_kind", "force") == "depth":
        inputs = np.stack([
            _live_depth_channel(frame.get("depth"), model.input_hw)
            for frame in (left, right)
        ], axis=0).astype(np.float32)
    else:
        left_force = np.asarray(left["force"], dtype=np.float32)
        right_force = np.asarray(right["force"], dtype=np.float32)
        inputs = np.concatenate((left_force, right_force), axis=-1).transpose(2, 0, 1)
        inputs = signed_log1p(inputs)
    # 共用 checkpoint 归一化；深度使用训练保存的尺度，不计算当前帧最大值。
    inputs = (
        (inputs - norm.input_mean[:, None, None])
        / norm.input_std[:, None, None]
    ).astype(np.float32)
    with torch.no_grad():
        output = model(torch.from_numpy(inputs).unsqueeze(0).to(device))
    return output.cpu().numpy()[0] * norm.target_std + norm.target_mean


def tactile_peaks(left: dict, right: dict) -> tuple[float, float]:
    return tuple(
        float(np.linalg.norm(np.asarray(item["force6d"]), axis=-1).max())
        for item in (left, right)
    )


def check_safety(args, force_reader: ForceReader, left: dict, right: dict) -> None:
    for name, frame in (("left", left), ("right", right)):
        wrench = np.asarray(frame["force6d"], dtype=np.float64)
        if np.any(np.abs(wrench[:3]) >= TACTILE_FORCE_SAFETY_LIMIT_N):
            raise ForceLimitStop(f"{name} tactile force limit exceeded: {wrench[:3]}")
        if np.any(np.abs(wrench[3:]) >= TACTILE_TORQUE_SAFETY_LIMIT_NMM):
            raise ForceLimitStop(f"{name} tactile torque limit exceeded: {wrench[3:]}")
    if args.use_force_safety:
        raw_force_torque = force_reader.read()
        wrench = np.asarray(raw_force_torque, dtype=np.float64).reshape(6)
        wrench = wrench - np.asarray(
            getattr(args, "force_bias", np.zeros(6)), dtype=np.float64
        )
        if np.any(np.abs(wrench[:3]) >= args.force_hard_stop_n):
            raise ForceLimitStop(
                f"{args.force_source} force limit exceeded: {wrench[:3]}"
            )
        if np.any(np.abs(wrench[3:]) >= args.torque_hard_stop_nm):
            raise ForceLimitStop(
                f"{args.force_source} torque limit exceeded: {wrench[3:]}"
            )


def calibrate_force_bias(args, force_reader: ForceReader) -> np.ndarray:
    if not args.use_force_safety:
        return np.zeros(6)
    print(
        f"Taring {args.force_source} wrench for {FORCE_TARE_SECONDS:g} seconds; "
        "keep the robot stationary and unloaded..."
    )
    samples = []
    sample_count = max(10, int(round(FORCE_TARE_SECONDS * args.control_hz)))
    for _ in range(sample_count):
        raw_force_torque = force_reader.read()
        samples.append(np.asarray(raw_force_torque, dtype=np.float64).reshape(6))
        time.sleep(1.0 / args.control_hz)
    bias = np.mean(np.stack(samples), axis=0)
    print(f"{args.force_source} wrench bias:", np.round(bias, 4))
    return bias

# 增加安全打开函数
def open_gripper_after_force_stop(args, robot: ZMQClientRobot) -> None:
    if not args.execute or not args.open_gripper_on_force_stop:
        return

    try:
        # 当前 RG2 接口：0.0=完全打开，1.0=完全闭合。
        robot.command_gripper(0.0, force=args.gripper_force)
        print("Force protection: gripper open command sent.")
    except Exception as exc:
        # 不让夹爪通信异常覆盖最初的超力异常。
        print(f"Force protection: failed to open gripper: {exc}")


def command_pose(robot, target: np.ndarray, execute: bool) -> None:
    if execute:
        robot.command_tcp_pose(np.asarray(target, dtype=np.float64))


def move_pose(
    args,
    robot,
    tactile,
    force_reader,
    target: np.ndarray,
    duration: float,
) -> None:
    start = np.asarray(robot.get_tcp_pose(), dtype=np.float64)
    steps = max(1, int(round(duration * args.control_hz)))
    period = 1.0 / args.control_hz
    for index in range(1, steps + 1):
        tick = time.monotonic()
        left, right = tactile.read()
        check_safety(args, force_reader, left, right)
        command = interpolate_pose(start, target, smoothstep5(index / steps))
        command_pose(robot, command, args.execute)
        time.sleep(max(0.0, period - (time.monotonic() - tick)))


def predict_valid_offset(args, model, norm, tactile, device):
    left, right = tactile.read()
    peaks = tactile_peaks(left, right)
    if min(peaks) < args.min_tactile_peak:
        raise SafetyStop(f"insufficient bilateral tactile contact: peaks={peaks}")
    prediction = predict_offset(model, norm, left, right, device)
    return prediction.astype(np.float64), peaks, left, right


def infer_standard_pose_from_current(current_tcp: np.ndarray, offset: np.ndarray):
    """Return T_base_reference = T_base_current @ inverse(T_reference_current)."""
    current_R = rotvec_to_matrix(current_tcp[3:])
    offset_R = rpy_to_matrix(offset[3:])
    standard_R = current_R @ offset_R.T
    standard_p = current_tcp[:3] - standard_R @ offset[:3]
    return standard_p, standard_R


def full_cycle_grasp(args, robot, tactile, force_reader, model, norm, device):
    grasp = np.asarray(args.grasp_tcp_pose, dtype=np.float64)
    approach = grasp.copy()
    approach[2] += args.approach_height_mm / 1000.0
    if args.execute:
        robot.command_gripper(0.0, force=args.gripper_force)
        print(
            f"Waiting {args.gripper_settle_s:.1f}s for the open gripper to settle "
            f"before re-taring the {args.force_source} wrench..."
        )
        time.sleep(args.gripper_settle_s)
        args.force_bias = calibrate_force_bias(args, force_reader)
    move_pose(args, robot, tactile, force_reader, approach, args.move_duration_s)
    move_pose(args, robot, tactile, force_reader, grasp, args.contact_duration_s)
    if args.execute:
        robot.command_gripper(1.0, force=args.gripper_force)
    time.sleep(args.gripper_settle_s)
    offset, peaks, left, right = predict_valid_offset(
        args, model, norm, tactile, device
    )
    check_safety(args, force_reader, left, right)
    print(
        "Grasp offset xyz(mm),rpy(deg):",
        np.round(np.r_[offset[:3] * 1000.0, np.rad2deg(offset[3:])], 3),
        "peaks=", peaks,
    )

    current = np.asarray(robot.get_tcp_pose(), dtype=np.float64)
    lift = current.copy()
    lift[2] += args.transport_height_mm / 1000.0
    move_pose(args, robot, tactile, force_reader, lift, args.move_duration_s)

    contact_standard = np.asarray(args.bolt_contact_tcp_pose, dtype=np.float64)
    standard_R = rotvec_to_matrix(contact_standard[3:])
    offset_R = rpy_to_matrix(offset[3:])
    contact_actual = np.r_[
        contact_standard[:3] + standard_R @ offset[:3],
        matrix_to_rotvec(standard_R @ offset_R),
    ]
    above = contact_actual.copy()
    above[:3] += args.approach_height_mm / 1000.0 * args.bolt_axis_base
    transfer = above.copy()
    transfer[2] = max(lift[2], above[2])
    move_pose(args, robot, tactile, force_reader, transfer, args.move_duration_s)
    move_pose(args, robot, tactile, force_reader, above, args.move_duration_s)
    move_pose(args, robot, tactile, force_reader, contact_actual, args.contact_duration_s)
    return contact_standard[:3], standard_R, offset


def screw_only_initialization(args, robot, tactile, force_reader, model, norm, device):
    current = np.asarray(robot.get_tcp_pose(), dtype=np.float64)
    offset, peaks, left, right = predict_valid_offset(
        args, model, norm, tactile, device
    )
    check_safety(args, force_reader, left, right)
    standard_p, standard_R = infer_standard_pose_from_current(current, offset)
    print(
        "Initial predicted TCP offset xyz(mm),rpy(deg):",
        np.round(np.r_[offset[:3] * 1000.0, np.rad2deg(offset[3:])], 3),
        "peaks=", peaks,
    )
    return standard_p, standard_R, offset


def fuse_pose_predictions(predictions: list[np.ndarray]) -> np.ndarray:
    """Average xyz directly and average RPY through projected rotation matrices."""
    values = np.asarray(predictions, dtype=np.float64)
    position = values[:, :3].mean(axis=0)
    rotation_sum = sum(
        (rpy_to_matrix(value[3:]) for value in values),
        np.zeros((3, 3), dtype=np.float64),
    )
    u, _, vt = np.linalg.svd(rotation_sum)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    # Return RPY using the same ZYX convention as the training labels.
    pitch = np.arctan2(
        -rotation[2, 0],
        np.hypot(rotation[2, 1], rotation[2, 2]),
    )
    rpy = np.array(
        [
            np.arctan2(rotation[2, 1], rotation[2, 2]),
            pitch,
            np.arctan2(rotation[1, 0], rotation[0, 0]),
        ]
    )
    return np.r_[position, rpy]


def relative_pose_offset(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return ``inverse(T_reference) @ T_current`` for xyz + RPY poses."""
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    reference_R = rpy_to_matrix(reference[3:])
    current_R = rpy_to_matrix(current[3:])
    relative_R = reference_R.T @ current_R
    relative_p = reference_R.T @ (current[:3] - reference[:3])
    return np.r_[relative_p, matrix_to_rpy(relative_R)]


def filter_pose_prediction(
    previous: np.ndarray,
    current: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Low-pass filter xyz + RPY without averaging Euler angles directly."""
    previous = np.asarray(previous, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    previous_R = rpy_to_matrix(previous[3:])
    current_R = rpy_to_matrix(current[3:])
    filtered_R = previous_R @ rotvec_to_matrix(
        alpha * rotation_vector(previous_R.T @ current_R)
    )
    filtered_p = (1.0 - alpha) * previous[:3] + alpha * current[:3]
    return np.r_[filtered_p, matrix_to_rpy(filtered_R)]


def tcp_from_local_offset(standard_tcp: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Compose a standard base-frame TCP pose with a local xyz + RPY offset."""
    standard_tcp = np.asarray(standard_tcp, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    standard_R = rotvec_to_matrix(standard_tcp[3:])
    offset_R = rpy_to_matrix(offset[3:])
    return np.r_[
        standard_tcp[:3] + standard_R @ offset[:3],
        matrix_to_rotvec(standard_R @ offset_R),
    ]


def observe_placement_until_enter(
    args, tactile, force_reader, model, norm, device, close_returned_at: float,
) -> None:
    """Print raw predictions at up to 5 Hz without changing the baseline or TCP."""
    print(
        "[placement test] Observation only: printing raw predictions at up to "
        "5 Hz; the original baseline is unchanged. Manually perturb the nut "
        "if needed, release it and keep clear, then press ENTER to start servo.",
        flush=True,
    )
    while True:
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if readable:
            if sys.stdin.readline() == "":
                raise EOFError("Standard input closed before placement servo confirmation")
            return

        tick = time.monotonic()
        prediction, peaks, left, right = predict_valid_offset(
            args, model, norm, tactile, device
        )
        check_safety(args, force_reader, left, right)
        values = np.r_[prediction[:3] * 1000.0, np.rad2deg(prediction[3:])]
        print(
            f"[placement test] observe t={time.monotonic() - close_returned_at:.2f}s "
            "since close command returned: prediction xyz(mm),rpy(deg)="
            f"{np.round(values, 3)} peaks=({peaks[0]:.4f},{peaks[1]:.4f})",
            flush=True,
        )
        # Wake promptly on ENTER instead of sleeping for the entire interval.
        select.select(
            [sys.stdin], [], [], max(0.0, 0.2 - (time.monotonic() - tick))
        )


def run_grasp_placement_test(
    args, robot, tactile, force_reader, model, norm, device,
) -> None:
    """Servo the grasped nut toward its pre-perturbation world pose.

    The pre-perturbation prediction is treated as the zero point. During
    correction, every tactile frame produces a baseline-relative offset and a
    bounded TCP command anchored at the standard pose. Predictions are never
    accumulated as incremental motion commands.
    """
    standard_tcp = np.asarray(args.grasp_tcp_pose, dtype=np.float64)
    approach_tcp = standard_tcp.copy()
    approach_tcp[2] += args.approach_height_mm / 1000.0

    print("[placement test] Opening RG2 and moving to the standard grasp pose.")
    if args.execute:
        robot.command_gripper(0.0, force=args.gripper_force)
    time.sleep(args.gripper_settle_s)
    if args.use_force_safety:
        args.force_bias = calibrate_force_bias(args, force_reader)
    move_pose(
        args, robot, tactile, force_reader,
        approach_tcp, args.move_duration_s,
    )
    move_pose(
        args, robot, tactile, force_reader,
        standard_tcp, args.contact_duration_s,
    )

    print("[placement test] TCP is stationary; closing RG2 now.")
    if args.execute:
        robot.command_gripper(1.0, force=args.gripper_force)
    close_returned_at = time.monotonic()
    time.sleep(args.gripper_settle_s)
    baseline, baseline_peaks, left, right = predict_valid_offset(
        args, model, norm, tactile, device
    )
    check_safety(args, force_reader, left, right)
    print(
        "[placement test] Prediction before manual perturbation "
        "xyz(mm),rpy(deg)=",
        np.round(
            np.r_[baseline[:3] * 1000.0, np.rad2deg(baseline[3:])], 3
        ),
        "peaks=", baseline_peaks,
    )

    observe_placement_until_enter(
        args, tactile, force_reader, model, norm, device, close_returned_at
    )

    period = 1.0 / args.control_hz
    max_steps = None
    if args.placement_test_servo_duration_s > 0.0:
        max_steps = max(
            1,
            int(round(args.placement_test_servo_duration_s * args.control_hz)),
        )
    filtered_prediction = baseline.copy()
    final_offset = np.zeros(6, dtype=np.float64)

    if max_steps is None:
        print(
            f"[placement test] Starting continuous real-time tactile servo at "
            f"{args.control_hz:g} Hz. Press Ctrl+C to stop; RG2 remains closed."
        )
    else:
        print(
            f"[placement test] Starting timed real-time tactile servo: "
            f"{max_steps} steps at {args.control_hz:g} Hz. RG2 remains closed."
        )

    step = 0
    while max_steps is None or step < max_steps:
        tick = time.monotonic()
        prediction, peaks, left, right = predict_valid_offset(
            args, model, norm, tactile, device
        )
        check_safety(args, force_reader, left, right)
        filtered_prediction = filter_pose_prediction(
            filtered_prediction,
            prediction,
            args.servo_filter_alpha,
        )
        # The pre-perturbation grasp is the zero point for this test. This
        # removes static model bias, so no perturbation means no motion.
        offset = relative_pose_offset(baseline, filtered_prediction)
        offset_R = rpy_to_matrix(offset[3:])
        offset_angle = float(np.linalg.norm(rotation_vector(offset_R)))
        offset_position = float(np.linalg.norm(offset[:3]))
        # if offset_position > args.placement_test_max_position_mm / 1000.0:
            # raise SafetyStop(
            #     "placement-test prediction exceeds position limit: "
            #     f"{offset_position * 1000.0:.3f} mm"
            # )
        # if offset_angle > np.deg2rad(args.placement_test_max_rotation_deg):
        #     raise SafetyStop(
        #         "placement-test prediction exceeds rotation limit: "
        #         f"{np.rad2deg(offset_angle):.3f} deg"
        #     )

        target_tcp = tcp_from_local_offset(standard_tcp, offset)
        current_tcp = np.asarray(robot.get_tcp_pose(), dtype=np.float64)
        command = limit_pose_step(
            current_tcp,
            target_tcp,
            args.max_command_position_step_mm / 1000.0,
            np.deg2rad(args.max_command_rotation_step_deg),
        )
        command_pose(robot, command, args.execute)
        final_offset = offset

        is_last_step = max_steps is not None and step == max_steps - 1
        if step % args.placement_test_print_every == 0 or is_last_step:
            progress = (
                f"{step + 1:03d}/{max_steps}"
                if max_steps is not None
                else f"{step + 1:06d}"
            )
            print(
                f"[placement test] servo {progress}: "
                "prediction xyz(mm),rpy(deg)="
                f"{np.round(np.r_[prediction[:3] * 1000.0, np.rad2deg(prediction[3:])], 3)} "
                "relative offset="
                f"{np.round(np.r_[offset[:3] * 1000.0, np.rad2deg(offset[3:])], 3)} "
                "command dp(mm)="
                f"{np.round((command[:3] - current_tcp[:3]) * 1000.0, 3)} "
                f"peaks=({peaks[0]:.4f},{peaks[1]:.4f})",
                flush=True,
            )
        step += 1
        time.sleep(max(0.0, period - (time.monotonic() - tick)))

    final_tcp = np.asarray(robot.get_tcp_pose(), dtype=np.float64)
    final_prediction, final_peaks, left, right = predict_valid_offset(
        args, model, norm, tactile, device
    )
    check_safety(args, force_reader, left, right)
    print("[placement test] final TCP [m,rotvec]=", np.round(final_tcp, 6))
    print(
        "[placement test] final relative offset xyz(mm),rpy(deg)=",
        np.round(
            np.r_[final_offset[:3] * 1000.0, np.rad2deg(final_offset[3:])], 3
        ),
    )
    print(
        "[placement test] final tactile prediction xyz(mm),rpy(deg)=",
        np.round(
            np.r_[
                final_prediction[:3] * 1000.0,
                np.rad2deg(final_prediction[3:]),
            ],
            3,
        ),
        "peaks=", final_peaks,
    )
    print(
        "[placement test] Complete. RG2 remains closed so you can inspect "
        "whether the nut returned to its standard world pose."
    )


def run_screw(
    args, robot, tactile, force_reader, model, norm, device,
    contact_p, contact_R, initial_offset,
):
    period = 1.0 / args.control_hz
    duration = abs(args.turns) * 60.0 / args.screw_rpm
    steps = max(1, int(round(duration * args.control_hz)))
    total_angle = args.rotation_sign * abs(args.turns) * 2.0 * np.pi
    total_descent = abs(args.turns) * args.screw_pitch_mm / 1000.0
    initial_offset_R = rpy_to_matrix(initial_offset[3:])
    filtered_dp = np.zeros(3)
    filtered_drot = np.zeros(3)

    for index in range(steps + 1):
        tick = time.monotonic()
        phase = index / steps
        theta = total_angle * phase
        turns = abs(theta) / (2.0 * np.pi)
        # axis_rotation 生成绕指定轴旋转指定角度的旋转矩阵
        path_R = axis_rotation(args.bolt_axis_base, theta) @ contact_R
        path_p = contact_p - total_descent * phase * args.bolt_axis_base
        left, right = tactile.read()
        check_safety(args, force_reader, left, right)
        peaks = tactile_peaks(left, right)

        prediction = None
        servo_on = turns >= args.servo_start_turns
        if servo_on and min(peaks) >= args.min_tactile_peak:
            prediction = predict_offset(model, norm, left, right, device).astype(np.float64)
            position_error = args.servo_ref[:3] - prediction[:3]
            reference_R = rpy_to_matrix(args.servo_ref[3:])
            predicted_R = rpy_to_matrix(prediction[3:])
            rotation_error = rotation_vector(reference_R @ predicted_R.T)
            target_dp = clipped(
                args.servo_kp[:3] * position_error,
                args.max_servo_position_mm / 1000.0,
            )
            target_drot = clipped(
                args.servo_kp[3:] * rotation_error,
                np.deg2rad(args.max_servo_rotation_deg),
            )
            # Never let tactile feedback alter rotation about the bolt axis.
            bolt_axis_local = path_R.T @ args.bolt_axis_base
            target_drot -= bolt_axis_local * np.dot(
                target_drot, bolt_axis_local
            )
            alpha = args.servo_filter_alpha
            filtered_dp = (1.0 - alpha) * filtered_dp + alpha * target_dp
            filtered_drot = (1.0 - alpha) * filtered_drot + alpha * target_drot

        desired_p = path_p + path_R @ (initial_offset[:3] + filtered_dp)
        desired_R = path_R @ rotvec_to_matrix(filtered_drot) @ initial_offset_R
        desired = np.r_[desired_p, matrix_to_rotvec(desired_R)]
        current = np.asarray(robot.get_tcp_pose(), dtype=np.float64)
        command = limit_pose_step(
            current,
            desired,
            args.max_command_position_step_mm / 1000.0,
            np.deg2rad(args.max_command_rotation_step_deg),
        )
        command_pose(robot, command, args.execute)

        if index % args.screw_print_every == 0:
            # filtered_dp is expressed in the rotating path/tool frame. Project
            # it into the base-frame bolt axis for an intuitive axial correction.
            servo_dp_mm = filtered_dp * 1000.0
            axis_mm = float(
                np.dot(path_R @ filtered_dp, args.bolt_axis_base) * 1000.0
            )
            advance_mm = total_descent * phase * 1000.0
            print(
                f"screw {index}/{steps} "
                f"theta={np.rad2deg(theta):+.1f}deg "
                f"axis={axis_mm:.3f}mm "
                f"advance={advance_mm:+.3f}mm "
                f"peaks=({peaks[0]:.3f},{peaks[1]:.3f}) "
                f"prediction = ({np.round(prediction[:3] * 1000.0, 3)},{np.round(np.rad2deg(prediction[3:]), 3)})"
                "servo_dp="
                f"{np.array2string(servo_dp_mm, precision=4, floatmode='fixed')}mm",
                flush=True,
            )
        time.sleep(max(0.0, period - (time.monotonic() - tick)))


def release_and_retreat(args, robot, tactile, force_reader):
    if args.execute:
        robot.command_gripper(0.0, force=args.gripper_force)
    time.sleep(args.gripper_settle_s)
    current = np.asarray(robot.get_tcp_pose(), dtype=np.float64)
    retreat = current.copy()
    retreat[:3] += args.retreat_height_mm / 1000.0 * args.bolt_axis_base
    move_pose(args, robot, tactile, force_reader, retreat, args.move_duration_s)


def validation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    cos_theta = (np.trace(rotation) - 1.0) / 2.0
    theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    if theta < 1e-12:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        diag = np.diag(rotation)
        axis_index = int(np.argmax(diag))
        axis = np.zeros(3, dtype=np.float64)
        axis[axis_index] = np.sqrt(max(0.0, (diag[axis_index] + 1.0) / 2.0))
        if axis[axis_index] < 1e-8:
            axis[axis_index] = 1.0
        for idx in range(3):
            if idx == axis_index:
                continue
            axis[idx] = (
                rotation[idx, axis_index] + rotation[axis_index, idx]
            ) / (4.0 * axis[axis_index])
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return np.zeros(3)
        return axis / norm * theta
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * np.sin(theta)
    return axis * theta


def sample_validation_tcp(reference: np.ndarray) -> np.ndarray:
    """Sample a TCP pose whose fingertip center stays on the sampled nut center.

    Position noise is applied to the calibrated nut-center pose ``N``. The TCP
    orientation is sampled independently, then its position is recovered from
    the fixed calibrated TCP-to-nut-center lever arm. At zero noise and zero
    rotation offset this reconstructs ``CALIBRATED_TCP_POSE_BASE`` exactly.

    This changes only reset-pose generation. Recorded fields and the saved
    pose label remain relative to ``CALIBRATED_TCP_POSE_BASE`` so existing and
    newly collected episodes keep the same schema and label convention.
    """
    # Snapshot of the collection L-reset settings; label reference is separate.
    CALIBRATED_TCP_POSE_BASE = np.asarray(reference, dtype=np.float64)
    NUT_CENTER_POSE_BASE = CALIBRATED_TCP_POSE_BASE.copy()
    # 修改：与采集当前实际参数一致，旋转中心在标定 TCP 下方 100 mm。
    # NUT_CENTER_POSE_BASE[2] -= 0.280
    NUT_CENTER_POSE_BASE[2] -= 0.10
    RESET_TCP_POSITION_RANDOM_RANGE_X = 0.000
    # 修改：同步采集的基座 Y/Z 平移范围和局部 X 旋转范围。
    # RESET_TCP_POSITION_RANDOM_RANGE_Y = 0.000
    # RESET_TCP_POSITION_RANDOM_RANGE_Z = 0.000
    # RESET_TCP_X_ROTATION_RANGE_DEG = 8.0
    RESET_TCP_POSITION_RANDOM_RANGE_Y = 0.001
    RESET_TCP_POSITION_RANDOM_RANGE_Z = 0.005
    RESET_TCP_X_ROTATION_RANGE_DEG = 5.0
    RESET_TCP_Y_ROTATION_MIN_DEG = 0.2
    # 修改：局部 Y 旋转幅度与采集一致，为 0.2～0.4 度，再随机正负。
    # RESET_TCP_Y_ROTATION_RANGE_DEG = 0.5
    RESET_TCP_Y_ROTATION_RANGE_DEG = 0.4
    # 修改：与采集的指尖坐标系一致，使用标定 TCP 的朝向。
    # reset_rotation_vector = np.array([0.0, np.pi, 0.0])
    reset_rotation_vector = CALIBRATED_TCP_POSE_BASE[3:6].copy()

    def rotation_matrix(angle, direction):
        return axis_rotation(np.asarray(direction, dtype=np.float64), angle)

    target = np.zeros(6, dtype=np.float64)

    # 先在基座坐标系中采样螺母中心 N，而不是直接平移 TCP。
    sampled_nut_center = NUT_CENTER_POSE_BASE[:3].copy()
    position_ranges = np.asarray(
        [
            RESET_TCP_POSITION_RANDOM_RANGE_X,
            RESET_TCP_POSITION_RANDOM_RANGE_Y,
            RESET_TCP_POSITION_RANDOM_RANGE_Z,
        ],
        dtype=np.float64,
    )
    sampled_nut_center += np.random.uniform(-position_ranges, position_ranges)

    # L 随机复位姿态：先取得基础 TCP 姿态。
    base_rotation = rotvec_to_matrix(reset_rotation_vector)
    # 右乘 X 轴增量，因此绕 TCP 自身 X 轴随机旋转。
    tcp_x_rotation = rotation_matrix(
        np.random.uniform(
            -np.deg2rad(RESET_TCP_X_ROTATION_RANGE_DEG),
            np.deg2rad(RESET_TCP_X_ROTATION_RANGE_DEG),
        ),
        (1.0, 0.0, 0.0),
    )
    # 继续右乘 Y 轴增量，因此绕变化后的 TCP 自身 Y 轴随机旋转。
    tcp_y_rotation_deg = np.random.uniform(
        RESET_TCP_Y_ROTATION_MIN_DEG,
        RESET_TCP_Y_ROTATION_RANGE_DEG,
    ) * np.random.choice((-1.0, 1.0))
    tcp_y_rotation = rotation_matrix(
        np.deg2rad(tcp_y_rotation_deg),
        (0.0, 1.0, 0.0),
    )
    target_rotation = (
        base_rotation
        @ tcp_x_rotation
        # 修改：与采集一致，先右乘局部 X，再右乘局部 Y 旋转。
        @ tcp_y_rotation
    )
    target[3:6] = validation_matrix_to_rotvec(target_rotation)

    # 在标定姿态中求 TCP -> 螺母中心的工具坐标系固定向量（当前为
    # [0, 0, 0.100] m，使用默认标定姿态时）。姿态改变后反算 TCP 位置，使这一杠杆端点始终
    # 落在 sampled_nut_center，从而围绕螺母中心转动而不是原地转 TCP。
    calibrated_rotation = rotvec_to_matrix(CALIBRATED_TCP_POSE_BASE[3:6])
    tcp_to_nut_center_tcp = calibrated_rotation.T @ (
        NUT_CENTER_POSE_BASE[:3] - CALIBRATED_TCP_POSE_BASE[:3]
    )
    target[:3] = sampled_nut_center - target_rotation @ tcp_to_nut_center_tcp
    return target


def validation_pose_target(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Collection label: inverse(T_reference_tcp) @ T_actual_tcp, xyz + RPY."""
    reference_R = rotvec_to_matrix(reference[3:])
    return np.r_[
        reference_R.T @ (actual[:3] - reference[:3]),
        matrix_to_rpy(reference_R.T @ rotvec_to_matrix(actual[3:])),
    ]


def run_pose_validation(args, robot, tactile, force_reader, model, norm, device):
    """Random open/move/close cycles; predictions never command the arm."""
    if not args.execute:
        return
    period = 1.0 / args.control_hz
    reference = np.asarray(args.validation_reference_tcp, dtype=np.float64)
    reset_reference = np.asarray(args.validation_reset_reference_tcp, dtype=np.float64)
    if args.validation_seed is not None:
        np.random.seed(args.validation_seed)
    print("[validation] Label reference TCP [m,rotvec]:", reference)
    print("[validation] Reset reference TCP [m,rotvec]:", reset_reference)
    # 修改：打印实际用于复位的标定朝向，避免仍显示旧的固定值。
    print("[validation] Reset base rotation vector:", reset_reference[3:6])

    def checked_read():
        left, right = tactile.read()
        check_safety(args, force_reader, left, right)
        return left, right

    def joints():
        value = np.asarray(robot.get_observations()["joint_positions"], dtype=np.float64)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise SafetyStop("Invalid joint feedback during pose validation")
        return value

    def set_gripper(value):
        deadline = time.monotonic() + args.validation_timeout_s
        while True:
            checked_read()
            robot.command_gripper(value, force=args.gripper_force)
            # Same actual open/closed feedback threshold as collection.
            if abs(float(joints()[-1]) - value) <= 0.03:
                return
            if time.monotonic() >= deadline:
                raise SafetyStop("Timed out waiting for validation gripper feedback")
            time.sleep(period)

    valid = 0
    for index in range(args.validation_rounds):
        set_gripper(0.0)
        target_tcp = sample_validation_tcp(reset_reference)
        current = joints()
        target_joints = np.asarray(
            robot.inverse_kinematics(target_tcp, qnear=current[:6]), dtype=np.float64
        ).reshape(-1)
        if target_joints.size < 6 or not np.all(np.isfinite(target_joints)):
            raise SafetyStop("Invalid IK result during pose validation")
        print(f"[validation {index + 1}/{args.validation_rounds}] Random TCP:", target_tcp)
        deadline = time.monotonic() + args.validation_timeout_s
        while True:
            tick = time.monotonic()
            checked_read()
            current = joints()
            delta = target_joints[:6] - current[:6]
            max_delta = float(np.max(np.abs(delta)))
            reached = max_delta < 0.005
            if time.monotonic() >= deadline:
                raise SafetyStop("Timed out reaching validation joint target")
            command = current.copy()
            if max_delta > args.validation_max_joint_step_rad:
                command[:6] += delta / max_delta * args.validation_max_joint_step_rad
            else:
                command[:6] = target_joints[:6]
            # Collection sends the final joint target and closes on the arrival frame.
            command[-1] = 1.0 if reached else 0.0
            robot.command_joint_state(command)
            if reached:
                robot.command_gripper(1.0, force=args.gripper_force)
                break
            time.sleep(max(0.0, period - (time.monotonic() - tick)))

        set_gripper(1.0)
        # Collection starts its 0.5 s timer only after confirmed closure.
        sample_at = time.monotonic() + args.validation_close_wait_s
        while time.monotonic() < sample_at:
            checked_read()
            time.sleep(min(period, max(0.0, sample_at - time.monotonic())))
        left, right = checked_read()
        actual = np.asarray(robot.get_tcp_pose(), dtype=np.float64)
        if actual.shape != (6,) or not np.all(np.isfinite(actual)):
            raise SafetyStop("Invalid TCP feedback during pose validation")
        if min(tactile_peaks(left, right)) < args.min_tactile_peak:
            print("[validation] Invalid round: no bilateral contact; skipped.")
            continue
        prediction = predict_offset(model, norm, left, right, device)
        target = validation_pose_target(actual, reference)
        if not np.all(np.isfinite(prediction)):
            raise SafetyStop("Nonfinite validation prediction")
        error = prediction - target
        error[3:] = np.arctan2(np.sin(error[3:]), np.cos(error[3:]))
        print("                 prediction        target           error")
        for i, name in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
            scale = 1000.0 if i < 3 else 180.0 / np.pi
            unit = "mm" if i < 3 else "deg"
            print(f"{name:>5s} ({unit}): {prediction[i] * scale:+11.4f} "
                  f"{target[i] * scale:+11.4f} {error[i] * scale:+11.4f}")
        valid += 1
    set_gripper(0.0)
    print(f"[validation] Complete: {valid}/{args.validation_rounds} valid rounds; gripper open.")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, norm = load_model(args.checkpoint, device)
    robot = ZMQClientRobot(
        port=args.robot_port, host=args.host, timeout_ms=args.robot_timeout_ms
    )
    tactile = None
    force_reader = None
    try:
        tactile = TactileSource(args)
        if robot.num_dofs() != 7:
            raise RuntimeError("expected a 7-DoF UR5 + RG2 robot server")
        if args.use_force_safety:
            force_reader = ForceReader(args, robot)
            force_reader.read()
        # Warm up every source before asking for execution confirmation.
        current = np.asarray(robot.get_tcp_pose(), dtype=np.float64)
        args.force_bias = calibrate_force_bias(args, force_reader)
        left, right = tactile.read()
        peaks = tactile_peaks(left, right)
        check_safety(args, force_reader, left, right)
        print("Current TCP [m,rotvec]:", np.round(current, 6))
        if min(peaks) >= args.min_tactile_peak:
            offset = predict_offset(model, norm, left, right, device)
            print(
                "Current tactile prediction [mm,deg]:",
                np.round(np.r_[offset[:3] * 1000.0, np.rad2deg(offset[3:])], 3),
                "peaks=", peaks,
            )
        else:
            print(
                "No bilateral contact during warm-up; model inference deferred. "
                f"peaks={peaks}"
            )
            if args.mode == "screw-only" and not args.grasp_placement_test:
                raise SafetyStop("screw-only mode requires an already grasped nut")
        if not args.execute:
            print("Observation-only check complete. Add --execute to command the robot.")
            return
        if not args.yes:
            answer = input(
                "Real UR5 motion will start. Keep the emergency stop ready. Type RUN: "
            ).strip()
            if answer != "RUN":
                raise SystemExit("Cancelled without sending a motion command")

        if args.mode == "validate-pose":
            run_pose_validation(args, robot, tactile, force_reader, model, norm, device)
            return
        if args.grasp_placement_test:
            run_grasp_placement_test(
                args, robot, tactile, force_reader, model, norm, device
            )
            return
        if args.mode == "full-cycle":
            contact_p, contact_R, initial_offset = full_cycle_grasp(
                args, robot, tactile, force_reader, model, norm, device
            )
        else:
            contact_p, contact_R, initial_offset = screw_only_initialization(
                args, robot, tactile, force_reader, model, norm, device
            )
        run_screw(
            args, robot, tactile, force_reader, model, norm, device,
            contact_p, contact_R, initial_offset,
        )
        release_and_retreat(args, robot, tactile, force_reader)
        print("Tactile-servo screw sequence completed.")
    except KeyboardInterrupt:
        print("Interrupted; no further motion commands will be sent.")
    except ForceLimitStop as exc:
        print(f"FORCE LIMIT STOP: {exc}. No further arm commands will be sent.")
        open_gripper_after_force_stop(args, robot)
        raise
    except SafetyStop as exc:
        print(f"SAFETY STOP: {exc}. No further motion commands will be sent.")
        raise
    finally:
        with ExitStack() as cleanup:
            cleanup.callback(robot.close)
            if tactile is not None:
                cleanup.callback(tactile.close)
            if force_reader is not None:
                cleanup.callback(force_reader.close)


if __name__ == "__main__":
    main()
