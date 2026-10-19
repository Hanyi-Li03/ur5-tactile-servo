import threading
import time

import numpy as np


XENSE_DEFAULT_LEFT_SERIAL = "OG001783"
XENSE_DEFAULT_RIGHT_SERIAL = "OG001782"
XENSE_DEFAULT_FORCE_HEIGHT = 35
XENSE_DEFAULT_FORCE_WIDTH = 20
_XENSE_SDK_LOCK = threading.Lock()


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
        self.device_label = (
            f"cam_id={self.cam_id}"
            if self.cam_id is not None
            else f"serial={self.serial_number}"
        )
        try:
            with _XENSE_SDK_LOCK:
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
        return format_xense_tactile_frame(
            {
                "force": force,
                "force6d": force6d,
                "depth": depth,
                "rectify": rectify,
                "timestamp": sensor_timestamp,
                "host_timestamp": time.time(),
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
                now = time.monotonic()
                if now - self._last_error_print > 2.0:
                    print(f"{self._name} async read failed: {exc}")
                    self._last_error_print = now
                time.sleep(0.02)
                continue

            with self._lock:
                self._latest = self._copy_frame(frame)

            if self._poll_interval_s > 0.0:
                self._stop_event.wait(self._poll_interval_s)

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

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        close = getattr(self._reader, "close", None)
        if callable(close):
            close()
