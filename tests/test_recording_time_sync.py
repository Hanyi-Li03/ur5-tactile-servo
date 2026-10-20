"""Exercise time pairing and the collector with simulated I/O only."""
from collections import deque
import json
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gello.robots.rtde_timing import RTDEClockMap
from gello.robots.ur import URRobot
from gello.tactile import xense
from gello.zmq_core import tactile_node
from real_ur5rg2 import collect_lerobot_xenstactile_servo_control as collect
from real_ur5rg2.tactile_time_sync import (
    JointTactileSynchronizer, TactileSyncUnavailable, save_episode_with_tactile_timing,
)


@pytest.fixture
def clock(monkeypatch):
    result = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(collect.time, "time", lambda: result.now)
    monkeypatch.setattr(collect.time, "monotonic", lambda: result.now)
    monkeypatch.setattr(collect.time, "sleep", lambda delay: setattr(result, "now", result.now + delay))
    return result


def state(timestamp, value=1, native=None, consistent=True):
    return {
        "observation_timestamp": timestamp,
        "joint_positions": np.full(7, value, dtype=float),
        "tcp_pose": np.array([value * .001, -.5, .137, 0, np.pi, 0]),
        "rtde_timestamp": timestamp if native is None else native,
        "rtde_timestamp_consistent": consistent,
        "rtde_clock_epoch": 0,
        "observation_time_basis": "ur_rtde_clock_mapped",
    }


def frame(timestamp, value=1):
    return {
        "capture_timestamp": timestamp, "timestamp": timestamp + .08,
        "host_timestamp": timestamp + .09, "valid": True,
        "force": np.full((35, 20, 3), value, dtype=np.float32),
        "force6d": np.full(6, value, dtype=np.float32),
        "depth": np.full((35, 20), value, dtype=np.float32),
        "rectify": np.zeros((8, 8, 3), dtype=np.uint8),
    }


class Tactile:
    def __init__(self, *timestamps):
        self.frames = [frame(value) for value in timestamps]

    def recording_timestamps(self, after=None):
        return [value["capture_timestamp"] for value in self.frames
                if after is None or value["capture_timestamp"] > after]

    def read_at(self, target, window, after=None):
        possible = [value for value in self.frames if after is None or value["capture_timestamp"] > after]
        if not possible:
            return None
        result = min(possible, key=lambda value: abs(value["capture_timestamp"] - target))
        return result if abs(result["capture_timestamp"] - target) <= window else None


def test_joint_reference_uses_matching_history_and_same_tcp(clock):
    sync = JointTactileSynchronizer()
    clock.now += .10
    sync.add_state(state(1000.025, 2))
    sync.add_state(state(1000.090, 9))
    left, right = Tactile(1000.027), Tactile(1000.031)
    observation, frames = sync.select(left, right)
    assert observation["observation_timestamp"] == 1000.025
    assert observation["tcp_pose"][0] == .002
    np.testing.assert_array_equal(observation["joint_positions"], np.full(7, 2))
    assert [value["capture_timestamp"] for value in frames] == [1000.027, 1000.031]


@pytest.mark.parametrize("offset", [-.025, .025])
def test_outside_window_is_not_replaced_by_host_receipt_time(clock, offset):
    sync = JointTactileSynchronizer()
    clock.now += .1
    sync.add_state(state(1000.05))
    left, right = Tactile(1000.05 + offset), Tactile(1000.05)
    left.frames[0]["host_timestamp"] = 1000.05
    with pytest.raises(TactileSyncUnavailable, match="within 20 ms"):
        sync.select(left, right)
    assert sync.cursors == [None, None]


def test_commit_prevents_reusing_either_capture_and_reset_clears_episode(clock):
    sync = JointTactileSynchronizer()
    clock.now += .1
    sync.add_state(state(1000.04))
    left, right = Tactile(1000.045), Tactile(1000.046)
    observation, frames = sync.select(left, right)
    sync.commit(observation, frames, 0)
    sync.add_state(state(1000.055, 2))
    with pytest.raises(TactileSyncUnavailable, match="unused captures"):
        sync.select(left, right)
    sync.reset()
    assert not sync.rows and not sync.states and sync.cursors == [None, None]
    with pytest.raises(TactileSyncUnavailable):
        sync.select(left, right)


def test_same_rtde_packet_deduplicated_and_inconsistent_snapshot_rejected(clock):
    sync = JointTactileSynchronizer()
    clock.now += .1
    sync.add_state(state(1000.03, native=12.0))
    sync.add_state(state(1000.030001, native=12.0))
    sync.add_state(state(1000.05, native=12.01, consistent=False))
    assert len(sync.states) == 1
    observation = state(1000.06, native=1.0)
    observation["rtde_clock_epoch"] = 1
    sync.add_state(observation)
    assert len(sync.states) == 1 and sync.states[0]["rtde_timestamp"] == 1.0


def test_returned_frame_revalidated_after_rpc(clock):
    sync = JointTactileSynchronizer()
    clock.now += .1
    sync.add_state(state(1000.05))
    left, right = Tactile(1000.05), Tactile(1000.05)
    right.read_at = lambda *_: frame(1000.09)
    with pytest.raises(TactileSyncUnavailable, match="outside window"):
        sync.select(left, right)


@pytest.fixture
def reader(monkeypatch):
    thread = SimpleNamespace(start=lambda: None, join=lambda **_: None)
    monkeypatch.setattr(xense.threading, "Thread", lambda **_: thread)
    return xense.AsyncXenseTactileReader(SimpleNamespace())


def test_capture_history_deduplicates_reprocessed_image_and_selects_nearest(clock, reader):
    reader._publish_frame(frame(1000.0, 1))
    # SDK TimeStamp/host receipt may change while the underlying capture is the same.
    reader._publish_frame(frame(1000.0, 2))
    clock.now += .03
    reader._publish_frame(frame(1000.03, 3))
    assert reader.recording_timestamps() == [1000.0, 1000.03]
    matched = reader.read_at(1000.004)
    np.testing.assert_array_equal(matched["force"], np.ones((35, 20, 3)))
    assert reader.read_at(1000.1) is None
    assert reader.recording_timestamps(1000.0) == [1000.03]
    matched["force"][:] = 99
    assert reader.read_at(1000.004)["force"][0, 0, 0] == 1


def test_incompatible_future_clock_fails_explicitly(clock, reader):
    reader._publish_frame(frame(clock.now + 1.0))
    with pytest.raises(RuntimeError, match="host clock"):
        reader.recording_timestamps()


@pytest.mark.parametrize("age_s", [.25, 1.0])
def test_delayed_captures_pair_to_historical_state_without_age_rejection(clock, reader, age_s):
    sync = JointTactileSynchronizer()
    clock.now += .05
    reference = clock.now
    sync.add_state(state(reference, 2))
    reader._publish_frame(frame(reference, .25))
    right = Tactile(reference + .008)
    clock.now += age_s
    # 已超过原先 150 ms 门限，仍能通过启动检查，并在保存时按时间差匹配。
    assert reader.wait_for_frame(.01)
    assert reader.recording_timestamps() == [reference]
    observation, frames = sync.select(reader, right)
    assert observation["observation_timestamp"] == reference
    assert observation["tcp_pose"][0] == .002
    assert frames[0]["capture_timestamp"] == reference
    assert frames[1]["capture_timestamp"] == reference + .008
    sync.commit(observation, frames, 0)
    assert reader.recording_timestamps(sync.cursors[0]) == []
    with pytest.raises(TactileSyncUnavailable, match="unused captures"):
        sync.select(reader, right)


def test_delayed_but_misaligned_captures_still_rejected(clock, reader):
    sync = JointTactileSynchronizer()
    clock.now += .05
    reference = clock.now
    sync.add_state(state(reference))
    reader._publish_frame(frame(reference + .025))
    clock.now += .5
    with pytest.raises(TactileSyncUnavailable, match="within 20 ms"):
        sync.select(reader, Tactile(reference))


def test_delayed_capture_before_episode_start_still_rejected(clock, reader):
    reader._publish_frame(frame(clock.now))
    clock.now += .5
    sync = JointTactileSynchronizer()
    clock.now += .001
    sync.add_state(state(clock.now))
    with pytest.raises(TactileSyncUnavailable, match="unused captures"):
        sync.select(reader, Tactile(clock.now))


def test_sdk_requests_capture_time_with_exact_force_depth_frame(monkeypatch, clock):
    outputs = SimpleNamespace(TimeStamp=1, ForceResultant=2, Force=3, Depth=4,
                              Rectify=5, CaptureTimeStamp=6)
    result = frame(clock.now)
    calls = []

    def select(*keys):
        calls.append(keys)
        return (result["timestamp"], result["force6d"], result["force"],
                result["depth"], result["rectify"], result["capture_timestamp"])

    sensor = SimpleNamespace(selectSensorInfo=select)
    monkeypatch.setitem(sys.modules, "xensesdk", SimpleNamespace(
        Sensor=SimpleNamespace(OutputType=outputs, create=lambda *_a, **_k: sensor)))
    actual = xense.XenseTactileReader("fake").read()
    assert calls == [(1, 2, 3, 4, 5, 6)]
    assert actual["capture_timestamp"] == clock.now
    assert actual["timestamp"] == clock.now + .08


def test_rtde_clock_fit_and_cached_packet_mapping():
    mapping = RTDEClockMap()
    for index in range(1500):
        elapsed = index * .008
        native = 10 + elapsed
        stamp = mapping.observe(native, 100 + elapsed + .003 + (index % 5) * .001)
    assert stamp is not None
    assert abs(stamp.monotonic - (100 + elapsed + .003)) < .001
    assert mapping.observe(native, 105 + elapsed) == stamp
    epoch = stamp.epoch
    assert mapping.observe(1.0, 200.0) is None
    assert mapping.epoch > epoch


def test_robot_snapshot_retries_rtde_packet_boundary_without_gripper_delay(clock):
    robot = URRobot.__new__(URRobot)
    native = iter([1.0, 1.008, 1.008, 1.008])
    reads = []
    robot.r_inter = SimpleNamespace(
        getTimestamp=lambda: next(native),
        getActualQ=lambda: reads.append("q") or np.arange(6, dtype=float),
        getActualTCPPose=lambda: np.arange(6, dtype=float) + 10,
        getActualTCPForce=lambda: np.zeros(6),
    )
    robot._use_gripper = True
    robot.last_gripper_scalar = .75
    robot._get_gripper_pos = lambda: pytest.fail("snapshot waits for gripper")
    robot._timestamp_tracker = SimpleNamespace(stamp=lambda _:
        SimpleNamespace(monotonic=clock.now - .008, epoch=0, drift_ppm=0, observed_jitter_s=0))
    observation = robot.get_observations()
    assert reads == ["q", "q"]
    assert observation["rtde_timestamp_consistent"]
    assert observation["observation_timestamp"] == clock.now - .008
    np.testing.assert_array_equal(observation["tcp_pose"], np.arange(6) + 10)
    assert observation["joint_positions"][-1] == .75


def test_tactile_rpc_routes_history_and_read_at(clock, reader):
    reader._publish_frame(frame(clock.now))
    server = tactile_node.ZMQServerTactile.__new__(tactile_node.ZMQServerTactile)
    server._tactile = reader
    server._stop_event = SimpleNamespace(is_set=lambda: not requests)
    requests = deque([
        {"method": "recording_timestamps", "after_capture_timestamp": 999.0},
        {"method": "read_at", "target_timestamp": 1000.0, "max_skew_s": .02},
        {"method": "read"},
    ])
    responses = []
    server._socket = SimpleNamespace(
        setsockopt=lambda *_: None,
        recv=lambda: pickle.dumps(requests.popleft()),
        send=lambda value: responses.append(pickle.loads(value)),
    )
    server.serve()
    assert responses[0] == [1000.0]
    assert responses[1]["capture_timestamp"] == 1000.0
    assert responses[2]["valid"]


def test_timing_saved_with_episode_and_schema_unchanged(clock, tmp_path):
    sync = JointTactileSynchronizer()
    clock.now += .1
    observation = state(1000.05)
    sync.commit(observation, [frame(1000.051), frame(1000.049)], 0)
    dataset = SimpleNamespace(root=tmp_path, meta=SimpleNamespace(total_episodes=5))
    dataset.save_episode = lambda: setattr(dataset.meta, "total_episodes", 6)
    save_episode_with_tactile_timing(dataset, sync)
    saved = json.loads((tmp_path / "meta/joint_tactile_timing/episode_000005.json").read_text())
    assert saved["frames"][0]["reference_timestamp"] == 1000.05
    np.testing.assert_allclose(saved["frames"][0]["capture_offset_ms"], [1, -1])
    assert not sync.rows


@pytest.mark.parametrize("episode_count", [1, 2])
def test_auto_collection_retries_missing_match_then_saves_one_matched_frame(clock, monkeypatch, tmp_path, episode_count):
    monkeypatch.setattr(sys, "argv", ["collect", "--no-collect-force", "--no-force-safety",
        "--no-preview", "--num-episodes", str(episode_count), "--root", str(tmp_path), "--image-size", "8"])
    dataset = SimpleNamespace(root=tmp_path, meta=SimpleNamespace(total_episodes=0), episode_buffer=None)
    saved, pending = [], []
    dataset.create_episode_buffer = lambda: pending
    dataset.clear_episode_buffer = pending.clear
    dataset.add_frame = lambda value, **_: pending.append(value)

    def save():
        saved.append(list(pending))
        pending.clear()
        dataset.meta.total_episodes += 1

    dataset.save_episode = save
    dataset.features = {key: {} for key in collect.TACTILE_DATASET_KEYS}
    for side in ("left", "right"):
        dataset.features[f"observation.tactile_{side}.depth"] = {"dtype": "image", "shape": (35, 20, 3)}

    class Robot:
        def __init__(self):
            self.gripper = 0.0
            self.gripper_commands = []

        def num_dofs(self):
            return 7

        def get_tcp_pose(self):
            pose = collect.CALIBRATED_TCP_POSE_BASE.copy()
            pose[2] += .012
            return pose

        def get_observations(self):
            clock.now += .001
            result = state(clock.now, 0)
            result["joint_positions"][-1] = self.gripper
            # 单独 get_tcp_pose() 故意返回不同的值，以发现标签误用了另一次反馈。
            result["tcp_pose"] = collect.CALIBRATED_TCP_POSE_BASE.copy()
            return result

        def inverse_kinematics(self, *_a, **_k):
            return np.zeros(6)

        def command_joint_state(self, _):
            pass

        def command_gripper(self, value, **_):
            self.gripper = value
            self.gripper_commands.append(value)

        def close(self):
            pass

    class LiveTactile(Tactile):
        def recording_timestamps(self, after=None):
            self.frames = [frame(clock.now - .001, .25)]
            return super().recording_timestamps(after)

        def read(self):
            return frame(clock.now, 1)

        def close(self):
            pass

    class Teleop:
        gripper_closed = False
        screw_direction = 0
        count = 0

        def poll(self, _):
            self.count += 1
            assert self.count < 100, "automatic saving stalled after skipping a match"
            event = "random_reset_pose" if self.count == 1 else None
            return np.zeros(3), np.eye(3), float(self.gripper_closed), False, event

        def close(self):
            pass

    class RetrySync(JointTactileSynchronizer):
        attempts = 0

        def select(self, *values):
            self.attempts += 1
            if self.attempts == 1:
                raise TactileSyncUnavailable("simulated missing capture")
            return super().select(*values)

    robot = Robot()
    camera = SimpleNamespace(read=lambda *_: (np.zeros((8, 8, 3), dtype=np.uint8), None))
    monkeypatch.setattr(collect, "ZMQClientRobot", lambda **_: robot)
    monkeypatch.setattr(collect, "ZMQClientCamera", lambda **_: camera)
    monkeypatch.setattr(collect, "ZMQClientTactile", lambda **_: LiveTactile())
    monkeypatch.setattr(collect, "KeyboardTeleop", Teleop)
    monkeypatch.setattr(collect, "PreviewWindow", lambda **_:
        SimpleNamespace(enabled=False, show=lambda *_: None, close=lambda: None))
    monkeypatch.setattr(collect, "make_dataset", lambda _: dataset)
    monkeypatch.setattr(collect, "JointTactileSynchronizer", RetrySync)
    collect.main()
    assert len(saved) == episode_count and all(len(episode) == 1 for episode in saved)
    assert robot.gripper_commands[-1] == 0.0
    row = saved[0][0]
    np.testing.assert_array_equal(row["observation.tactile_left.distributed_force"], np.full((35, 20, 3), .25))
    np.testing.assert_allclose(row["observation.fingertip_center_pose_relative_to_nut"], 0, atol=1e-6)
    audit = json.loads((tmp_path / "meta/joint_tactile_timing/episode_000000.json").read_text())
    assert len(audit["frames"]) == 1
    assert max(abs(value) for value in audit["frames"][0]["capture_offset_ms"]) <= 20
    if episode_count == 2:
        next_audit = json.loads((tmp_path / "meta/joint_tactile_timing/episode_000001.json").read_text())
        assert next_audit["frames"][0]["reference_timestamp"] > audit["frames"][0]["reference_timestamp"]
        assert next_audit["frames"][0]["frame_index"] == 0
