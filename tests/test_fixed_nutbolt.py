"""Check the fixed nut/bolt path with simulated I/O only."""

from contextlib import ExitStack
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tactile_policy import run_fixed_nutbolt_miben_lhy as entry


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=0.0)
    monkeypatch.setattr(entry.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(entry.time, "sleep", lambda seconds: setattr(state, "now", state.now + seconds))
    return state


class FakeRobot:
    def __init__(self, clock):
        self.clock = clock
        self.pose = entry.CALIBRATED_TCP_POSE_BASE.copy()
        self.events = []

    def get_tcp_pose(self):
        return self.pose.copy()

    def command_tcp_pose(self, pose):
        self.pose = np.asarray(pose).copy()
        self.events.append(("pose", self.clock.now, self.pose.copy()))

    def command_gripper(self, value, force):
        self.events.append(("gripper", self.clock.now, value, force))


def test_default_plan_does_not_connect_to_hardware(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fixed-screw"])
    monkeypatch.setattr(entry, "ZMQClientRobot", lambda **_: pytest.fail("plan opened robot"))
    monkeypatch.setattr(entry, "ZMQClientTactile", lambda **_: pytest.fail("plan opened Xense"))
    monkeypatch.setattr(entry, "RobotiqFTS300Stream", lambda **_: pytest.fail("plan opened FT300"))
    entry.main()
    assert "no hardware connection" in capsys.readouterr().out


def test_fixed_grasp_transport_pause_screw_and_release(clock, monkeypatch):
    args = entry.parse_args(["--execute", "--turns", "0.01"])
    robot = FakeRobot(clock)
    safety = SimpleNamespace(check=lambda: None, tare=lambda: None,
                             require_grasp_contact=lambda: None)
    waypoints = []
    original_move = entry.move_pose

    def move(*values):
        waypoints.append(values[3].copy())
        original_move(*values)

    monkeypatch.setattr(entry, "move_pose", move)
    entry.full_cycle_grasp(args, robot, safety)
    np.testing.assert_allclose(waypoints[1], [-0.0635, -0.5499, 0.1372, 0, np.pi, 0])
    np.testing.assert_allclose(waypoints[2][:3], [-0.0635, -0.5499, 0.2172])
    np.testing.assert_allclose(waypoints[3][:3], [0.0365, -0.5474, 0.2172])
    np.testing.assert_allclose(waypoints[-1], [0.0365, -0.5474, 0.142, 0, np.pi, 0])
    for pose in waypoints:
        np.testing.assert_array_equal(pose[3:], [0, np.pi, 0])

    contact_command_time = robot.events[-1][1]
    assert clock.now - contact_command_time >= 1.0
    first_screw = len(robot.events)
    entry.run_screw(args, robot, safety)
    assert robot.events[first_screw][1] - contact_command_time >= 1.0
    screw_events = robot.events[first_screw:]
    positions = np.asarray([event[2][:3] for event in screw_events])
    np.testing.assert_allclose(positions[:, :2], np.tile([0.0365, -0.5474], (len(positions), 1)))
    assert np.all(np.diff(positions[:, 2]) <= 1e-12)
    np.testing.assert_allclose(positions[-1, 2], 0.142 - 0.01 * 0.0025)
    for _, _, pose in screw_events:
        np.testing.assert_allclose(entry.rotvec_to_matrix(pose[3:])[:, 2], [0, 0, -1], atol=1e-10)

    final_screw_pose = robot.pose.copy()
    entry.release_and_retreat(args, robot, safety)
    assert [event[2] for event in robot.events if event[0] == "gripper"] == [0.0, 1.0, 0.0]
    np.testing.assert_allclose(robot.pose[:3], final_screw_pose[:3] + [0, 0, 0.05])
    np.testing.assert_allclose(entry.rotvec_to_matrix(robot.pose[3:]),
                               entry.rotvec_to_matrix(final_screw_pose[3:]), atol=1e-10)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_screw_matches_original_nominal_path_for_multiple_turns(sign):
    args = entry.parse_args(["--rotation-sign", str(sign)])
    contact_rotation = entry.rotvec_to_matrix(np.array([0, np.pi, 0]))
    for phase in np.linspace(0, 1, 161):
        pose = entry.screw_pose(args, phase)
        theta = sign * 4 * 2 * np.pi * phase
        rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        np.testing.assert_allclose(entry.rotvec_to_matrix(pose[3:]), rz @ contact_rotation, atol=1e-12)
        np.testing.assert_allclose(pose[:3], [0.0365, -0.5474, 0.142 - 0.01 * phase])


def test_all_four_revolutions_are_commanded_with_a_vertical_axis(clock):
    args = entry.parse_args(["--execute", "--control-hz", "3"])
    robot = FakeRobot(clock)
    robot.pose = entry.BOLT_TCP_CONTACT_POSE.copy()
    safety = SimpleNamespace(check=lambda: None)
    entry.run_screw(args, robot, safety)
    rotations = np.array([entry.rotvec_to_matrix(event[2][3:]) for event in robot.events])
    relative = rotations[1:] @ rotations[:-1].transpose(0, 2, 1)
    increments = np.arctan2(relative[:, 1, 0], relative[:, 0, 0])
    np.testing.assert_allclose(increments.sum(), -4 * 2 * np.pi, atol=1e-10)
    assert np.all(increments < 0)
    assert np.all(np.abs(increments) <= np.deg2rad(0.5))
    np.testing.assert_allclose(rotations[:, :, 2], np.tile([0, 0, -1], (len(rotations), 1)), atol=1e-12)
    np.testing.assert_allclose(robot.pose[:3], [0.0365, -0.5474, 0.132])


@pytest.mark.parametrize("angle", [np.pi, np.pi - 1e-7, -np.pi + 1e-7])
def test_rotation_conversion_is_stable_near_the_downward_pose(angle):
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)
    rotation = entry.rotvec_to_matrix(axis * angle)
    np.testing.assert_allclose(entry.rotvec_to_matrix(entry.matrix_to_rotvec(rotation)), rotation, atol=1e-12)


def test_command_step_limits_bound_translation_and_rotation():
    args = entry.parse_args([])
    current = entry.BOLT_TCP_CONTACT_POSE.copy()
    desired = entry.screw_pose(args, 0.03)
    desired[0] += 0.1
    command = entry.limit_pose_step(args, current, desired)
    np.testing.assert_allclose(np.linalg.norm(command[:3] - current[:3]), 0.0005)
    relative = entry.rotvec_to_matrix(command[3:]) @ entry.rotvec_to_matrix(current[3:]).T
    np.testing.assert_allclose(np.linalg.norm(entry.matrix_to_rotvec(relative)), np.deg2rad(0.5))


@pytest.mark.parametrize("wrench,valid,error", [
    ([25, 0, 0, 0, 0, 0], True, entry.ForceLimitStop),
    ([0, 0, 0, 0, -100, 0], True, entry.ForceLimitStop),
    ([float("nan"), 0, 0, 0, 0, 0], True, entry.SafetyStop),
    ([0, 0, 0, 0, 0, 0], False, entry.SafetyStop),
    (None, True, entry.SafetyStop),
])
def test_invalid_or_excessive_tactile_force_stops_before_motion(monkeypatch, clock, wrench, valid, error):
    args = entry.parse_args(["--execute"])
    monkeypatch.setattr(entry, "ZMQClientTactile", lambda **_: SimpleNamespace(
        read=lambda: {"valid": valid, "force6d": wrench}, close=lambda: None))
    robot = FakeRobot(clock)
    with ExitStack() as resources:
        safety = entry.SafetyMonitor(args, robot, resources)
        with pytest.raises(error):
            entry.run_screw(args, robot, safety)
    assert robot.events == []


@pytest.mark.parametrize("left,right,has_contact", [
    (0, 0, False), (0.2, 0, False), (0, 0.2, False),
    (0.2, 0.3, True), (1e-4, 1e-4, True), (1e-5, 0.2, False),
])
def test_grasp_requires_bilateral_contact_without_inference(monkeypatch, clock, left, right, has_contact):
    args = entry.parse_args([])

    def tactile(port, **_):
        force = left if port == args.left_tactile_port else right
        return SimpleNamespace(read=lambda: {"valid": True, "force6d": [force, 0, 0, 0, 0, 0]},
                               close=lambda: None)

    monkeypatch.setattr(entry, "ZMQClientTactile", tactile)
    with ExitStack() as resources:
        safety = entry.SafetyMonitor(args, FakeRobot(clock), resources)
        if has_contact:
            safety.require_grasp_contact()
        else:
            with pytest.raises(entry.SafetyStop, match="insufficient bilateral tactile contact"):
                safety.require_grasp_contact()


def test_missing_grasp_contact_stops_before_lifting(monkeypatch, clock):
    robot = FakeRobot(clock)
    robot.close = lambda: None
    robot.num_dofs = lambda: 7
    frame = {"valid": True, "force6d": np.zeros(6)}
    monkeypatch.setattr(sys, "argv", ["fixed-screw", "--execute", "--yes"])
    monkeypatch.setattr(entry, "ZMQClientRobot", lambda **_: robot)
    monkeypatch.setattr(entry, "ZMQClientTactile", lambda **_: SimpleNamespace(
        read=lambda: frame, close=lambda: None))
    with pytest.raises(entry.SafetyStop, match="insufficient bilateral tactile contact"):
        entry.main()
    np.testing.assert_allclose(robot.pose, entry.CALIBRATED_TCP_POSE_BASE)
    assert robot.events[-1][0] == "gripper" and robot.events[-1][2] == 1.0


def test_main_force_stop_opens_gripper_without_further_arm_motion(monkeypatch, clock):
    robot = FakeRobot(clock)
    closed = []
    robot.close = lambda: closed.append("robot")
    robot.num_dofs = lambda: 7
    safe_frame = {"valid": True, "force6d": np.array([1, 0, 0, 0, 0, 0])}
    monkeypatch.setattr(sys, "argv", ["fixed-screw", "--execute", "--yes"])
    monkeypatch.setattr(entry, "ZMQClientRobot", lambda **_: robot)
    monkeypatch.setattr(entry, "ZMQClientTactile", lambda **_: SimpleNamespace(
        read=lambda: safe_frame, close=lambda: closed.append("tactile")))

    def force_stop(*_):
        raise entry.ForceLimitStop("simulated force limit")

    monkeypatch.setattr(entry, "run_screw", force_stop)
    with pytest.raises(entry.ForceLimitStop):
        entry.main()
    assert robot.events[-1][0] == "gripper" and robot.events[-1][2] == 0.0
    np.testing.assert_allclose(robot.pose[:3], entry.BOLT_TCP_CONTACT_POSE[:3])
    assert closed == ["tactile", "tactile", "robot"]


@pytest.mark.parametrize("flag,value", [
    ("--turns", "nan"), ("--screw-rpm", "inf"), ("--control-hz", "0"),
    ("--screw-pitch-mm", "-1"), ("--gripper-settle-s", "-1"),
    ("--screw-rpm", "200"), ("--min-tactile-peak", "0"),
])
def test_invalid_motion_parameters_are_rejected(flag, value):
    with pytest.raises(SystemExit) as exc:
        entry.parse_args([flag, value])
    assert exc.value.code == 2
