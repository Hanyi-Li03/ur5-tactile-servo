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
    """Keep SDK calls exclusive without letting one polling thread overtake another."""

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
    """Bound Xense model threads while keeping up with two 30 Hz sensors.

    xensesdk 2.1 does not expose SessionOptions. Its models are created
    synchronously by Sensor.create(), under _XENSE_SDK_LOCK. Restore the
    constructor afterwards and leave sessions created by other threads alone.
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
                "inside an environment with the vendor Xense SDK installed."
            ) from exc

        self.serial_number = serial_number
        self.cam_id = cam_id
        self.force_height = int(force_height)
        self.force_width = int(force_width)
        self.name = name
        self._Sensor = Sensor
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
                "capture_timestamp": capture_timestamp,
                "valid": (
                    force is not None and force6d is not None
                    and _float_or_zero(capture_timestamp) > 0
                ),
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
        max_age_s: float = 0.15,
    ) -> None:
        if not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("Xense max_age_s must be finite and positive")
        self._reader = tactile_reader
        self._poll_interval_s = max(0.0, float(poll_interval_s))
        self._name = name
        self._fallback = empty_xense_tactile_frame(force_height, force_width)
        self._latest = None
        self._history = deque(maxlen=4)
        self._capture_key = None
        self._last_progress = 0.0
        self._max_age_s = float(max_age_s)
        self._error = None
        self._closed = False
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
                if not bool(frame.get("valid", False)):
                    raise RuntimeError("Xense returned an invalid tactile frame")
                if not all(np.isfinite(frame[key]).all() for key in ("force", "force6d")):
                    raise RuntimeError("Xense returned non-finite tactile force")
                # TimeStamp can change when the SDK reprocesses the same image.
                capture_key = _float_or_zero(frame.get("capture_timestamp"))
                if capture_key <= 0:
                    capture_key = _float_or_zero(frame.get("timestamp"))
                if not np.isfinite(capture_key) or capture_key <= 0:
                    raise RuntimeError("Xense returned an invalid capture timestamp")
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                    self._history.clear()
                now = time.monotonic()
                if now - self._last_error_print > 2.0:
                    print(f"{self._name} async read failed: {exc}")
                    self._last_error_print = now
                time.sleep(0.02)
                continue

            with self._lock:
                if self._capture_key is not None and capture_key < self._capture_key:
                    self._error = "Xense capture timestamp went backwards; restart the node"
                    self._history.clear()
                elif capture_key != self._capture_key:
                    self._latest = self._copy_frame(frame)
                    self._capture_key = capture_key
                    self._last_progress = time.monotonic()
                    self._error = None
                    self._history.append((capture_key, self._latest, self._last_progress))

            if self._poll_interval_s > 0.0:
                self._stop_event.wait(self._poll_interval_s)

    def wait_for_frame(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            with self._lock:
                if self._error is None and self._latest is not None:
                    try:
                        self._check_fresh()
                    except RuntimeError:
                        pass
                    else:
                        return True
            time.sleep(0.01)
        return False

    def read(self) -> dict[str, np.ndarray | float | bool]:
        with self._lock:
            self._check_fresh()
            if self._latest is None:
                return self._copy_frame(self._fallback)
            return self._copy_frame(self._latest)

    def _check_fresh(self) -> None:
        """Called under the cache lock; repeated captures do not renew this age."""
        if self._closed:
            raise RuntimeError(f"{self._name} is closed")
        if self._error is not None:
            raise RuntimeError(f"{self._name} read failed: {self._error}")
        if self._latest is not None:
            age = time.monotonic() - self._last_progress
            if age > self._max_age_s:
                raise RuntimeError(f"{self._name} capture stopped updating ({age * 1000:.0f} ms)")
            captured = _float_or_zero(self._latest.get("capture_timestamp"))
            if captured > 0:
                capture_age = time.time() - captured
                if capture_age > self._max_age_s:
                    raise RuntimeError(f"{self._name} capture is too old ({capture_age * 1000:.0f} ms)")

    def read_for_recording(self, after_capture_timestamp: float | None = None,
                           target_timestamp: float | None = None) -> dict:
        """Return current safety values and at most one earlier capture for recording.

        The cursor belongs to the caller, so one client cannot consume another
        client's history. An empty result never waits for a future sensor frame.
        """
        if after_capture_timestamp is not None and not np.isfinite(after_capture_timestamp):
            raise ValueError("Recording cursor must be finite")
        if target_timestamp is not None and not np.isfinite(target_timestamp):
            raise ValueError("Recording target time must be finite")
        with self._lock:
            self._check_fresh()
            if self._latest is None:
                return {"latest": self._copy_frame(self._fallback), "recording": None}
            latest = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self._latest.items()
                if key in ("force6d", "valid", "timestamp", "host_timestamp", "capture_timestamp")
            }
            candidate = None
            candidates = []
            # Retain one capture of slack, not a growing FIFO backlog. Large
            # capture gaps cannot turn an old result into a one-frame buffer.
            for key, frame, published in list(self._history)[-2:]:
                if self._capture_key - key > 0.05:
                    continue
                if time.monotonic() - published > self._max_age_s:
                    continue
                captured = _float_or_zero(frame.get("capture_timestamp"))
                if captured > 0 and time.time() - captured > self._max_age_s:
                    continue
                if after_capture_timestamp is None or key > after_capture_timestamp:
                    candidates.append((key, frame))
            if candidates:
                selected = candidates[0] if target_timestamp is None else min(
                    candidates, key=lambda item: abs(item[0]-target_timestamp))
                candidate = self._copy_frame(selected[1])
            return {"latest": latest, "recording": candidate}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError(f"{self._name} worker did not stop; exit the hardware node")
        close = getattr(self._reader, "close", None)
        if callable(close):
            close()
