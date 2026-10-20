"""Map RTDE controller time onto the host monotonic clock without robot control."""
from collections import deque, OrderedDict
from dataclasses import dataclass
import math
import threading
import time

import numpy as np


@dataclass(frozen=True)
class RTDETime:
    native: float
    monotonic: float
    first_seen_monotonic: float
    epoch: int
    drift_ppm: float
    observed_jitter_s: float


class RTDEClockMap:
    """Fit the lower arrival envelope, retaining unknown minimum transport delay.

    Native timestamps are measured on the controller. A receive-only connection
    cannot determine the constant one-way network delay, so the mapped time is
    explicitly an estimate. Cached per-packet times never change after delivery.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.epoch = -1
        self._reset()

    def _reset(self):
        self.epoch += 1
        self.raw = deque(maxlen=8192)
        self.cache = OrderedDict()
        self.last_native = None
        self.last_progress = None
        self.origin = 0.0
        self.offset = 0.0
        self.slope = 0.0
        self.last_fit = -math.inf
        self.jitter_s = 0.0
        self.last_mapped = -math.inf

    def observe(self, native: float, received: float) -> RTDETime | None:
        if not all(math.isfinite(x) for x in (native, received)) or native <= 0:
            raise RuntimeError("Invalid RTDE controller timestamp")
        with self._lock:
            if self.last_native is not None and native < self.last_native-.5:
                self._reset()
            if native in self.cache:
                return self.cache[native]
            if self.last_native is not None and native <= self.last_native:
                return None
            if not self.raw:
                self.origin = native
            self.raw.append((native, received))
            while self.raw and received-self.raw[0][1] > 30:
                self.raw.popleft()
            self.last_native, self.last_progress = native, received
            if len(self.raw) < 16 or native-self.raw[0][0] < .1:
                return None
            if received-self.last_fit >= .25:
                x = np.array([row[0]-self.origin for row in self.raw])
                offsets = np.array([row[1]-row[0] for row in self.raw])
                slope = 0.0
                if x[-1]-x[0] >= 8:
                    best = {}
                    for index, value in enumerate(x):
                        key = int(value*2)
                        if key not in best or offsets[index] < offsets[best[key]]:
                            best[key] = index
                    ids = list(best.values())
                    bx, by = x[ids], offsets[ids]
                    dx = bx-bx.mean()
                    denominator = float(np.dot(dx, dx))
                    if denominator > 0:
                        slope = float(np.dot(dx, by-by.mean())/denominator)
                        slope = float(np.clip(slope, -.001, .001))
                self.slope = slope
                self.offset = float(np.min(offsets-slope*x))
                residual = offsets-(self.offset+slope*x)
                self.jitter_s = float(np.percentile(residual, 95))
                self.last_fit = received
            mapped = native+self.offset+self.slope*(native-self.origin)
            # The native time cannot be mapped later than its first observation.
            mapped = min(mapped, received)
            if mapped <= self.last_mapped:
                # A large improvement in the arrival floor invalidates earlier
                # clock calibration. Rewarm rather than publishing reversed time.
                self._reset()
                self.origin = native
                self.raw.append((native, received))
                self.last_native, self.last_progress = native, received
                return None
            stamp = RTDETime(native, mapped, received, self.epoch,
                             self.slope*1e6, self.jitter_s)
            self.last_mapped = mapped
            self.cache[native] = stamp
            if len(self.cache) > 2048:
                self.cache.popitem(last=False)
            return stamp


class RTDETimestampTracker:
    def __init__(self, receiver, poll_interval_s: float = .0005):
        self.receiver = receiver
        self.clock = RTDEClockMap()
        self._stop = threading.Event()
        self._error = None
        self._started = time.monotonic()
        self._interval = poll_interval_s
        self._thread = threading.Thread(target=self._run, name="ur-rtde-clock", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                native = self.receiver.getTimestamp()
                self.clock.observe(native, time.monotonic())
                self._error = None
            except Exception as exc:
                self._error = str(exc)
            self._stop.wait(self._interval)

    def stamp(self, native: float) -> RTDETime | None:
        if self._error is not None:
            raise RuntimeError(f"UR timestamp tracking failed: {self._error}")
        now = time.monotonic()
        progress = self.clock.last_progress
        if progress is not None and now-progress > .1:
            raise RuntimeError("UR state timestamps stopped updating for more than 100 ms")
        if progress is None and now-self._started > 1:
            raise RuntimeError("UR has not provided a valid controller timestamp")
        return self.clock.observe(float(native), now)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
        if self._thread.is_alive():
            raise RuntimeError("UR timestamp tracking thread did not exit")
