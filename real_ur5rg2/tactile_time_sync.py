"""Pair tactile captures to actual UR state times, without changing robot control."""
from collections import deque
import json
from pathlib import Path
import time

import numpy as np


class TactileSyncUnavailable(RuntimeError):
    """This recording cycle has no complete joint/tactile match."""


class JointTactileSynchronizer:
    def __init__(self, max_skew_s=0.020):
        if not np.isfinite(max_skew_s) or not 0 < max_skew_s <= 0.05:
            raise ValueError("Tactile sync window must be in (0, 50] ms")
        self.max_skew_s = float(max_skew_s)
        self.states = deque(maxlen=32)
        self.rows = []
        # 跨 episode 限频，避免自动单帧采集时每次复位都立即刷日志。
        self._next_clock_print_at = 0.0
        self.reset()

    def reset(self):
        self.states.clear()
        self.rows.clear()
        self.cursors = [None, None]
        self.last_reference = None
        self.last_native = None
        self.clock_epoch = None
        # 新 episode 不使用开始录制之前捕获的触觉或关节反馈。
        self.started_at = time.time()

    def add_state(self, observation):
        if "observation_timestamp" not in observation or "tcp_pose" not in observation:
            raise RuntimeError("Robot node has no timed joint/TCP snapshot; restart launch_nodes.py")
        if not observation.get("rtde_timestamp_consistent", False):
            return
        epoch = observation.get("rtde_clock_epoch")
        if epoch != self.clock_epoch:
            self.states.clear()
            self.last_native = None
            self.clock_epoch = epoch
        native = float(observation["rtde_timestamp"])
        timestamp = float(observation["observation_timestamp"])
        if not all(np.isfinite(value) for value in (native, timestamp)) or timestamp <= 0:
            raise TactileSyncUnavailable("Invalid robot state timestamp")
        if self.last_native is not None:
            if native == self.last_native:
                return
            if native < self.last_native:
                self.states.clear()
                raise TactileSyncUnavailable("UR state clock moved backwards")
        if timestamp > time.time() + 0.005:
            raise TactileSyncUnavailable("UR mapped state time is ahead of the host clock")
        if self.states and timestamp <= self.states[-1]["observation_timestamp"]:
            self.states.clear()
            raise TactileSyncUnavailable("UR mapped state clock moved backwards")
        if (not np.isfinite(observation["joint_positions"]).all()
                or not np.isfinite(observation["tcp_pose"]).all()):
            raise TactileSyncUnavailable("Invalid joint/TCP snapshot")
        # 同时缓存 q 和 TCP；标签使用此 TCP，不在保存时重新查询另一个时刻。
        self.states.append({key: value.copy() if isinstance(value, np.ndarray) else value
                            for key, value in observation.items()})
        self.last_native = native

    def select(self, left, right):
        now = time.time()
        available = [client.recording_timestamps(cursor)
                     for client, cursor in zip((left, right), self.cursors)]
        # 在过滤/配对前打印，配不上时也能查看捕获时间。显示最近的候选捕获；
        # 没有新候选时显示上次使用的捕获。此日志只观察数值，不限制帧龄。
        tick = time.monotonic()
        if tick >= self._next_clock_print_at:
            local_time = time.time()
            for side, captures, cursor in zip(("left", "right"), available, self.cursors):
                captured = float(max(captures)) if captures else cursor
                values = ("capture_time=unavailable age_ms=unavailable" if captured is None else
                          f"capture_time={captured:.6f} age_ms={(local_time - captured) * 1000:.3f}")
                print(f"[tactile time] {side} local_time={local_time:.6f} {values}", flush=True)
            self._next_clock_print_at = tick + 1.0
        available = [[float(value) for value in times
                      if np.isfinite(value) and value >= self.started_at
                      and value <= now + 0.005]
                     for times in available]
        if not all(available):
            raise TactileSyncUnavailable("Waiting for unused captures from both tactile sensors")
        # 不限制距保存时刻的帧龄。历史样本只要来自本次录制、未使用，且左右
        # 捕获都能与同一 t_joint 配对，就可保存；时钟异常和去重检查继续保留。
        # 按关节状态从新到旧查找。触觉计算有延迟，因此允许选择短历史中的关节
        # 反馈，但参考始终是 t_joint，绝不将关节值替换成触觉时刻的插值或指令值。
        for observation in reversed(self.states):
            reference = float(observation["observation_timestamp"])
            if (reference < self.started_at or reference > now + 0.005
                    or (self.last_reference is not None and reference <= self.last_reference)):
                continue
            nearest = [min(times, key=lambda captured: abs(captured - reference))
                       for times in available]
            if any(abs(captured - reference) > self.max_skew_s for captured in nearest):
                continue
            frames = [client.read_at(reference, self.max_skew_s, cursor)
                      for client, cursor in zip((left, right), self.cursors)]
            # RPC 期间缓存可能更新/淘汰，再核对一次实际返回的帧，不能只信时间列表。
            for frame, cursor in zip(frames, self.cursors):
                if frame is None or not bool(frame.get("valid", False)):
                    raise TactileSyncUnavailable("Tactile capture changed before matching completed")
                captured = float(frame.get("capture_timestamp", 0.0))
                if (not np.isfinite(captured) or captured < self.started_at
                        or abs(captured - reference) > self.max_skew_s
                        or captured > time.time() + 0.005
                        or (cursor is not None and captured <= cursor)):
                    raise TactileSyncUnavailable("Returned tactile capture is invalid, reused, or outside window")
                if not all(np.isfinite(frame[key]).all() for key in ("force", "force6d", "depth")):
                    raise TactileSyncUnavailable("Non-finite tactile force/depth")
            return observation, frames
        raise TactileSyncUnavailable(
            f"No joint state with both tactile captures within {self.max_skew_s * 1000:.0f} ms")

    def commit(self, observation, frames, frame_index):
        # 仅在 dataset.add_frame 成功后推进游标，失败的配对不会消耗捕获。
        reference = float(observation["observation_timestamp"])
        captured = [float(frame["capture_timestamp"]) for frame in frames]
        self.last_reference = reference
        self.cursors = captured
        self.rows.append({
            "frame_index": int(frame_index),
            "reference_timestamp": reference,
            "reference_basis": observation["observation_time_basis"],
            "rtde_timestamp": float(observation["rtde_timestamp"]),
            "rtde_clock_epoch": observation.get("rtde_clock_epoch"),
            "rtde_clock_drift_ppm": observation.get("rtde_clock_drift_ppm"),
            "rtde_observed_jitter_ms": observation.get("rtde_observed_jitter_ms"),
            "capture_timestamp": captured,
            "capture_offset_ms": [(value - reference) * 1000 for value in captured],
            "left_right_delta_ms": abs(captured[0] - captured[1]) * 1000,
            "recorded_host_timestamp": time.time(),
        })


def save_episode_with_tactile_timing(dataset, synchronizer=None):
    dataset.save_episode()
    if synchronizer is None or not synchronizer.rows:
        return
    # 单独保存真实匹配时间，保留现有 Parquet 特征与旧数据集续采兼容。
    directory = Path(dataset.root) / "meta" / "joint_tactile_timing"
    directory.mkdir(parents=True, exist_ok=True)
    episode = dataset.meta.total_episodes - 1
    payload = {
        "reference": "UR joint state timestamp mapped to host clock",
        "max_skew_ms": synchronizer.max_skew_s * 1000,
        "notes": ("Tactile CaptureTimeStamp must share the host clock. "
                  "UR mapping retains unknown minimum network delay. "
                  "RGB and external force are not time-matched by this change."),
        "frames": synchronizer.rows,
    }
    path = directory / f"episode_{episode:06d}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    synchronizer.rows.clear()
