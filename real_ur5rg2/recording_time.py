"""Small histories used only when assembling recorded observations."""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import math
import time

import numpy as np


class RecordingAlignmentError(RuntimeError):
    pass


class RecordingSampleUnavailable(RecordingAlignmentError):
    """No matching candidate yet; distinct from a bad clock or sensor error."""


class RecordingWarmup(RecordingSampleUnavailable):
    """A valid capture predates the new episode's recording boundary."""


@dataclass(frozen=True)
class TimedSample:
    timestamp: float
    value: object
    timing: dict


class RecordingHistory:
    def __init__(self, name: str, maxlen: int = 16):
        self.name = name
        self.samples = deque(maxlen=maxlen)

    def clear(self):
        self.samples.clear()

    def add(self, timestamp: float, value, **timing):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp <= 0:
            raise RecordingAlignmentError(f"{self.name} has no usable sample time")
        if self.samples and timestamp < self.samples[-1].timestamp:
            self.clear()
            raise RecordingAlignmentError(f"{self.name} sample clock moved backwards")
        if not self.samples or timestamp != self.samples[-1].timestamp:
            self.samples.append(TimedSample(timestamp, value, timing))

    def nearest(self, target: float, max_skew_s: float, *, after=None) -> TimedSample:
        now = time.time()
        candidates = [sample for sample in self.samples
                      if (after is None or sample.timestamp > after)
                      and -0.005 <= now-sample.timestamp <= 0.15
                      and abs(sample.timestamp-target) <= max_skew_s]
        if not candidates:
            raise RecordingSampleUnavailable(f"{self.name} has no unused sample within {max_skew_s*1000:.0f} ms")
        return min(candidates, key=lambda sample: abs(sample.timestamp-target))

    def previous(self, target: float) -> TimedSample:
        candidates = [sample for sample in self.samples if 0 <= target-sample.timestamp <= 0.15]
        if not candidates:
            raise RecordingSampleUnavailable(f"{self.name} history is warming up")
        return candidates[-1]


class RecordingTimeAlignment:
    def __init__(self, max_skew_s: float = 0.020):
        if not math.isfinite(max_skew_s) or not 0 < max_skew_s <= 0.05:
            raise ValueError("Recording max skew must be in (0, 50] ms")
        self.max_skew_s = max_skew_s
        self.cameras = [RecordingHistory("base RGB"), RecordingHistory("wrist RGB")]
        self.states = RecordingHistory("robot state", maxlen=32)
        self.actions = RecordingHistory("gripper command", maxlen=32)
        self.reset()

    def reset(self):
        for history in [*self.cameras, self.states, self.actions]:
            history.clear()
        self.camera_cursors = [None, None]
        self.ur_clock_epoch = None
        self.last_ur_timestamp = None
        self.started_at = time.time()

    def add_state(self, observation):
        timestamp = observation.get("observation_timestamp")
        if timestamp is None:
            raise RecordingAlignmentError("Robot node has no state times; restart launch.py")
        epoch = observation.get("rtde_clock_epoch")
        if epoch != self.ur_clock_epoch:
            self.states.clear()
            self.ur_clock_epoch = epoch
            self.last_ur_timestamp = None
        if observation.get("rtde_timestamp_consistent", True):
            native_timestamp = observation.get("rtde_timestamp")
            if native_timestamp is not None:
                native_timestamp = float(native_timestamp)
                if not math.isfinite(native_timestamp) or native_timestamp < 0:
                    raise RecordingAlignmentError("Robot node has no usable RTDE packet time")
                if self.last_ur_timestamp is not None:
                    # Reading one RTDE packet twice can change the wall/monotonic
                    # conversion by microseconds. Keep its first history entry.
                    if native_timestamp == self.last_ur_timestamp:
                        return
                    if native_timestamp < self.last_ur_timestamp:
                        self.states.clear()
                        raise RecordingAlignmentError("robot state packet clock moved backwards")
            self.states.add(timestamp, np.asarray(observation["joint_positions"], dtype=np.float32).copy(),
                basis=observation.get("observation_time_basis", "host_state_read"),
                rtde_timestamp=observation.get("rtde_timestamp"), rtde_clock_epoch=epoch,
                rtde_clock_drift_ppm=observation.get("rtde_clock_drift_ppm"),
                observed_jitter_ms=observation.get("rtde_observed_jitter_ms"),
                tcp_pose=(np.asarray(observation["tcp_pose"], dtype=float).tolist()
                          if "tcp_pose" in observation else None),
                minimum_transport_delay_unknown=observation.get("rtde_minimum_transport_delay_unknown", True))
            self.last_ur_timestamp = native_timestamp

    def add_camera(self, index: int, frame: dict):
        if frame.get("timestamp_domain") not in ("timestamp_domain.global_time", "timestamp_domain.system_time"):
            raise RecordingAlignmentError("RGB has no host-compatible frame clock; restart/check camera node")
        self.cameras[index].add(frame["capture_timestamp"], frame["image"],
            basis="realsense_frame_global_time", frame_number=frame.get("frame_number"),
            received_timestamp=frame.get("host_timestamp"))

    def choose(self, target: float):
        if not math.isfinite(target) or target <= 0 or target > time.time()+0.005:
            raise RecordingAlignmentError("Recording histories are warming up or frame clock is invalid")
        if target < self.started_at:
            raise RecordingWarmup("Recording histories are warming up or frame clock is invalid")
        if time.time()-target > 0.15:
            raise RecordingAlignmentError("Recording reference is more than 150 ms old")
        cameras = [history.nearest(target, self.max_skew_s, after=cursor)
                   for history, cursor in zip(self.cameras, self.camera_cursors)]
        state = self.states.nearest(target, self.max_skew_s)
        command = self.actions.previous(target)
        return cameras, state, command

    def commit(self, cameras):
        self.camera_cursors = [sample.timestamp for sample in cameras]


def timing_entry(timestamp: float, reference: float, **details):
    return {"timestamp": float(timestamp), "offset_ms": (timestamp-reference)*1000, **details}


class RecordingPerformance:
    """Episode-only in-memory counters, written once with the saved episode."""
    def __init__(self, fps: float, force_crc_errors=None, *, preparing: bool = False):
        self.period_ms = 1000.0 / fps
        self.crc_at_start = force_crc_errors
        self.preparing = preparing
        self.preparation = ({
            "requested_host_timestamp": time.time(),
            "requested_monotonic": time.monotonic(),
            "waiting_loops": 0,
            "wait_reasons": Counter(),
            "condition_counts": Counter(),
            "recording_retry_counts": Counter(),
            "force_crc_errors_at_request": force_crc_errors,
        } if preparing else None)
        self._reset_loop_metrics()

    def _reset_loop_metrics(self):
        self.loops = 0
        self.saved_rows = 0
        self.last_loop = None
        self.intervals = []
        self.durations = defaultdict(list)
        self.skips = Counter()
        self.conditions = Counter()
        self.retries = Counter()

    def start_loop(self, started_monotonic: float):
        if self.preparing and self.loops:
            # The preceding loop never reached ready. Keep its diagnostics
            # separately; the first successful loop must retain all its stages.
            self.preparation["waiting_loops"] += self.loops
            self.preparation["wait_reasons"].update(self.skips)
            self.preparation["condition_counts"].update(self.conditions)
            self.preparation["recording_retry_counts"].update(self.retries)
            self._reset_loop_metrics()
        if self.last_loop is not None:
            self.intervals.append((started_monotonic-self.last_loop)*1000)
        self.last_loop = started_monotonic
        self.loops += 1

    def mark_ready(self, force_crc_errors=None) -> bool:
        """Start recording with the current valid row, preserving its metrics."""
        if not self.preparing:
            return False
        self.preparing = False
        ready_monotonic = time.monotonic()
        requested_crc = self.preparation["force_crc_errors_at_request"]
        self.preparation.update({
            "ready_host_timestamp": time.time(),
            "ready_monotonic": ready_monotonic,
            "elapsed_ms": (ready_monotonic-self.preparation["requested_monotonic"])*1000,
            "force_crc_errors_at_ready": force_crc_errors,
            "force_crc_errors_during_preparation": (force_crc_errors-requested_crc
                if force_crc_errors is not None and requested_crc is not None else None),
        })
        self.crc_at_start = force_crc_errors
        return True

    def duration(self, stage: str, started: float, finished=None):
        if finished is None:
            finished = time.perf_counter()
        self.durations[stage].append((finished-started)*1000)

    @staticmethod
    def _summary(values):
        if not values:
            return {"count": 0}
        return {"count": len(values), "mean_ms": float(np.mean(values)),
                "p95_ms": float(np.percentile(values, 95)), "max_ms": float(max(values))}

    def summary(self, force_crc_errors=None):
        intervals = self._summary(self.intervals)
        result = {
            "loops": self.loops, "saved_rows": self.saved_rows,
            "skip_reasons": dict(self.skips), "condition_counts": dict(self.conditions),
            "recording_retry_counts": dict(self.retries),
            "loop_intervals": intervals,
            "loop_hz": 1000/intervals["mean_ms"] if self.intervals else None,
            "loop_intervals_over_1_5_periods": sum(x > self.period_ms*1.5 for x in self.intervals),
            "stages": {stage: self._summary(values) for stage, values in self.durations.items()},
            "force_crc_errors_at_start": self.crc_at_start,
            "force_crc_errors_at_save": force_crc_errors,
            "force_crc_errors_during_episode": (force_crc_errors-self.crc_at_start
                if force_crc_errors is not None and self.crc_at_start is not None else None),
        }
        if self.preparation is not None:
            result["preparation"] = dict(self.preparation)
            for name in ("wait_reasons", "condition_counts", "recording_retry_counts"):
                result["preparation"][name] = dict(self.preparation[name])
        return result
