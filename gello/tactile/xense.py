import threading
import time
from collections import deque
from contextlib import contextmanager
from functools import wraps

import numpy as np


XENSE_DEFAULT_LEFT_SERIAL = "OG001783"
XENSE_DEFAULT_RIGHT_SERIAL = "OG001782"
XENSE_DEFAULT_FORCE_HEIGHT = 35
XENSE_DEFAULT_FORCE_WIDTH = 20
_XENSE_ONNX_INTRA_OP_THREADS = 6


class _XenseSDKLock:
    """按请求顺序串行执行 SDK 调用，防止一侧轮询连续抢锁。"""

    def __init__(self):
        self._condition = threading.Condition()
        self._queue = deque()
        self._held = False

    def acquire(self):
        ticket = object()
        with self._condition:
            self._queue.append(ticket)
            try:
                self._condition.wait_for(lambda: not self._held and self._queue[0] is ticket)
            except BaseException:
                self._queue.remove(ticket)
                self._condition.notify_all()
                raise
            self._queue.popleft()
            self._held = True
        return True

    def release(self):
        with self._condition:
            if not self._held:
                raise RuntimeError("Cannot release an unlocked Xense SDK lock")
            self._held = False
            self._condition.notify_all()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


_XENSE_SDK_LOCK = _XenseSDKLock()


@contextmanager
def _xense_onnx_threads():
    """复用参考仓库：Xense 会话内部 6 线程、会话间 1 线程，关闭自旋。

    仅在持有 SDK 锁的 Sensor.create() 阶段临时包装 ORT 构造函数。
    其他线程创建会话时保持原设置，创建完成或失败后均恢复原构造函数。
    """
    import onnxruntime as ort

    original_init = ort.InferenceSession.__init__
    creator_thread = threading.get_ident()
    sessions_created = 0

    @wraps(original_init)
    def session_init(session, path_or_bytes, sess_options=None, *args, **kwargs):
        nonlocal sessions_created
        is_xense_session = threading.get_ident() == creator_thread
        if is_xense_session:
            if sess_options is None:
                sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = _XENSE_ONNX_INTRA_OP_THREADS
            sess_options.inter_op_num_threads = 1
            sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            sess_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        original_init(session, path_or_bytes, sess_options, *args, **kwargs)
        if is_xense_session:
            sessions_created += 1

    ort.InferenceSession.__init__ = session_init
    try:
        yield
    finally:
        ort.InferenceSession.__init__ = original_init
        if sessions_created:
            print(
                f"Xense ONNX: {sessions_created} sessions, "
                f"intra_op_threads={_XENSE_ONNX_INTRA_OP_THREADS}, "
                "inter_op_threads=1, spinning=off"
            )


def empty_xense_tactile_frame(
    force_height: int = XENSE_DEFAULT_FORCE_HEIGHT,
    force_width: int = XENSE_DEFAULT_FORCE_WIDTH,
) -> dict[str, np.ndarray | float | bool]:
    return {
        "force": np.zeros((force_height, force_width, 3), dtype=np.float32),
        "force6d": np.zeros(6, dtype=np.float32),
        "depth": np.zeros((force_height, force_width), dtype=np.float32),
        "rectify": np.zeros((0, 0, 3), dtype=np.uint8),
        "timestamp": 0.0,
        "host_timestamp": 0.0,
        "capture_timestamp": 0.0,
        "valid": False,
    }


def _float_or_zero(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return 0.0


def _fit_force_map(
    value: object,
    force_height: int,
    force_width: int,
) -> tuple[np.ndarray, bool]:
    formatted = np.zeros((force_height, force_width, 3), dtype=np.float32)
    if value is None:
        return formatted, False

    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return formatted, False

    if array.size == 0:
        return formatted, False

    if array.ndim == 3 and array.shape[-1] == 3:
        height = min(array.shape[0], force_height)
        width = min(array.shape[1], force_width)
        formatted[:height, :width] = array[:height, :width, :3]
        return formatted, True

    try:
        flattened = array.reshape(-1, 3)
    except ValueError:
        return formatted, False

    count = min(flattened.shape[0], force_height * force_width)
    formatted.reshape(-1, 3)[:count] = flattened[:count]
    return formatted, count > 0


def _format_force6d(value: object) -> np.ndarray:
    if value is None:
        return np.zeros(6, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size < 6:
        array = np.pad(array, (0, 6 - array.size))
    return array[:6].astype(np.float32)


def format_xense_tactile_frame(
    frame: dict[str, object],
    force_height: int = XENSE_DEFAULT_FORCE_HEIGHT,
    force_width: int = XENSE_DEFAULT_FORCE_WIDTH,
) -> dict[str, np.ndarray | float | bool]:
    formatted = empty_xense_tactile_frame(force_height, force_width)
    force, force_ok = _fit_force_map(frame.get("force"), force_height, force_width)
    formatted["force"] = force
    formatted["force6d"] = _format_force6d(frame.get("force6d"))
    depth = frame.get("depth")
    if depth is not None:
        try:
            formatted["depth"] = np.asarray(depth, dtype=np.float32)
        except (TypeError, ValueError):
            pass
    rectify = frame.get("rectify")
    if rectify is not None:
        try:
            formatted["rectify"] = np.asarray(rectify)
        except (TypeError, ValueError):
            pass
    formatted["valid"] = bool(frame.get("valid", force_ok)) and force_ok
    formatted["timestamp"] = (
        _float_or_zero(frame.get("timestamp")) if formatted["valid"] else 0.0
    )
    formatted["host_timestamp"] = (
        _float_or_zero(frame.get("host_timestamp")) if formatted["valid"] else 0.0
    )
    formatted["capture_timestamp"] = (
        _float_or_zero(frame.get("capture_timestamp")) if formatted["valid"] else 0.0
    )
    return formatted


class XenseTactileReader:
    """Read one Xense tactile sensor through xensesdk."""

    def __init__(
        self,
        serial_number: str | None = None,
        *,
        cam_id: int | None = None,
        force_height: int = XENSE_DEFAULT_FORCE_HEIGHT,
        force_width: int = XENSE_DEFAULT_FORCE_WIDTH,
        name: str = "xense",
    ) -> None:
        if int(force_height) <= 0 or int(force_width) <= 0:
            raise ValueError("Xense force_height and force_width must be positive.")
        if serial_number is None and cam_id is None:
            raise ValueError("Pass either serial_number or cam_id for XenseTactileReader.")

        try:
            from xensesdk import Sensor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading Xense tactile sensors requires xensesdk. Run launch_nodes.py "
                "inside the environment where xense_record.py can import it."
            ) from exc

        self.serial_number = serial_number
        self.cam_id = cam_id
        self.force_height = int(force_height)
        self.force_width = int(force_width)
        self.name = name
        self._Sensor = Sensor
        if not hasattr(Sensor.OutputType, "CaptureTimeStamp"):
            raise RuntimeError("Xense time alignment requires SDK CaptureTimeStamp support")
        self.device_label = (
            f"cam_id={self.cam_id}"
            if self.cam_id is not None
            else f"serial={self.serial_number}"
        )
        try:
            with _XENSE_SDK_LOCK, _xense_onnx_threads():
                if self.cam_id is not None:
                    self._sensor = Sensor.create(cam_id=self.cam_id)
                else:
                    self._sensor = Sensor.create(self.serial_number)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open Xense {self.name} tactile sensor "
                f"{self.device_label}: {exc}"
            ) from exc
        self._outputs = (
            Sensor.OutputType.TimeStamp,
            Sensor.OutputType.ForceResultant,
            Sensor.OutputType.Force,
            Sensor.OutputType.Depth,
            Sensor.OutputType.Rectify,
            Sensor.OutputType.CaptureTimeStamp,
        )
        print(f"Xense {self.name} tactile ready: {self.device_label}")

    def read(self) -> dict[str, np.ndarray | float | bool]:
        try:
            with _XENSE_SDK_LOCK:
                values = self._sensor.selectSensorInfo(*self._outputs)
        except Exception as exc:
            raise RuntimeError(
                f"Xense {self.name} read failed for {self.device_label}: {exc}"
            ) from exc

        if not isinstance(values, tuple):
            values = (values,)
        if len(values) < 3:
            frame = empty_xense_tactile_frame(self.force_height, self.force_width)
            frame["host_timestamp"] = time.time()
            return frame

        sensor_timestamp, force6d, force = values[:3]
        depth = values[3] if len(values) > 3 else None
        rectify = values[4] if len(values) > 4 else None
        capture_timestamp = values[5] if len(values) > 5 else None
        return format_xense_tactile_frame(
            {
                "force": force,
                "force6d": force6d,
                "depth": depth,
                "rectify": rectify,
                "timestamp": sensor_timestamp,
                "host_timestamp": time.time(),
                # 与力、深度在同一次 SDK 查询中取得，避免混用不同图像的时间。
                "capture_timestamp": capture_timestamp,
                "valid": force is not None and force6d is not None,
            },
            self.force_height,
            self.force_width,
        )

    def close(self) -> None:
        release = getattr(self._sensor, "release", None)
        if callable(release):
            with _XENSE_SDK_LOCK:
                release()


class AsyncXenseTactileReader:
    def __init__(
        self,
        tactile_reader,
        *,
        force_height: int = XENSE_DEFAULT_FORCE_HEIGHT,
        force_width: int = XENSE_DEFAULT_FORCE_WIDTH,
        poll_interval_s: float = 0.0,
        name: str = "xense tactile",
    ) -> None:
        self._reader = tactile_reader
        self._poll_interval_s = max(0.0, float(poll_interval_s))
        self._name = name
        self._fallback = empty_xense_tactile_frame(force_height, force_width)
        self._latest = None
        # 按原始捕获去重。短缓存用于录制回看，read() 仍返回最新结果。
        self._history = deque(maxlen=16)
        self._capture_key = None
        self._history_error = None
        self._last_error_print = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _copy_frame(self, frame: dict[str, np.ndarray | float | bool]) -> dict:
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in frame.items()
        }

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._reader.read()
            except Exception as exc:
                with self._lock:
                    self._history_error = str(exc)
                    self._history.clear()
                now = time.monotonic()
                if now - self._last_error_print > 2.0:
                    print(f"{self._name} async read failed: {exc}")
                    self._last_error_print = now
                time.sleep(0.02)
                continue

            self._publish_frame(frame)

            if self._poll_interval_s > 0.0:
                self._stop_event.wait(self._poll_interval_s)

    def _publish_frame(self, frame):
        with self._lock:
            self._latest = self._copy_frame(frame)
            captured = _float_or_zero(frame.get("capture_timestamp"))
            if not np.isfinite(captured) or captured <= 0:
                self._history_error = "Xense has no valid CaptureTimeStamp; restart/check the SDK"
                self._history.clear()
            elif self._capture_key is not None and captured < self._capture_key:
                self._history_error = "Xense capture clock moved backwards; restart launch_nodes.py"
                self._history.clear()
            elif captured != self._capture_key:
                self._capture_key = captured
                self._history_error = None
                if bool(frame.get("valid", False)):
                    self._history.append((captured, self._copy_frame(frame)))

    def wait_for_frame(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None and bool(self._latest.get("valid", True)):
                    return True
            time.sleep(0.01)
        return False

    def read(self) -> dict[str, np.ndarray | float | bool]:
        with self._lock:
            if self._latest is None:
                return self._copy_frame(self._fallback)
            return self._copy_frame(self._latest)

    def _recording_candidates(self, after_capture_timestamp=None):
        """在缓存锁内调用；不等待未来帧，不用 host_timestamp 冒充捕获时间。"""
        if self._history_error is not None:
            raise RuntimeError(self._history_error)
        now = time.time()
        if self._capture_key is not None and self._capture_key > now + 0.005:
            raise RuntimeError("Xense CaptureTimeStamp is not on the host clock (future capture)")
        # 保存依据与关节参考时间的差值，不因 SDK 结果到达较晚而丢弃捕获。
        # 缓存长度和捕获游标仍限制回看范围；未来时间检查只用于发现时钟异常。
        return [(captured, frame) for captured, frame in self._history
                if captured <= now + 0.005
                and (after_capture_timestamp is None or captured > after_capture_timestamp)]

    def recording_timestamps(self, after_capture_timestamp=None) -> list[float]:
        # 先只传时间列表，避免为寻找参考关节状态传输整段图像历史。
        with self._lock:
            return [captured for captured, _ in self._recording_candidates(after_capture_timestamp)]

    def read_at(self, target_timestamp: float, max_skew_s: float = 0.020,
                after_capture_timestamp=None) -> dict | None:
        if not np.isfinite(target_timestamp) or not np.isfinite(max_skew_s) or max_skew_s <= 0:
            raise ValueError("Invalid tactile recording reference/window")
        with self._lock:
            candidates = self._recording_candidates(after_capture_timestamp)
            if not candidates:
                return None
            captured, frame = min(candidates, key=lambda item: abs(item[0] - target_timestamp))
            # |t_capture - t_joint| <= max_skew_s；否则本轮没有可保存的匹配帧。
            if abs(captured - target_timestamp) > max_skew_s:
                return None
            return self._copy_frame(frame)

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        close = getattr(self._reader, "close", None)
        if callable(close):
            close()
