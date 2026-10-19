"""Regression checks with simulated devices; never connect to physical hardware."""
from contextlib import ExitStack
import importlib
import json
import struct
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import zmq

from gello.cameras.lerobot_realsense_camera import LeRobotRealSenseCamera
from gello.cameras.realsense_camera import RealSenseCamera
from gello.force.robotiq_fts300 import AsyncForceReader, RobotiqFTS300ModbusSerial
from gello.force.robotiq_fts300_stream import FTStreamParser, modbus_crc
from gello.robots.rg2_gripper import AsyncRG2Gripper
from gello.robots.rtde_timing import RTDEClockMap
from gello.robots.ur import URRobot
from gello.tactile.xense import AsyncXenseTactileReader, _XenseSDKLock
from gello.zmq_core.camera_node import ZMQClientCamera, ZMQServerCamera
from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot
from gello.zmq_core.tactile_node import ZMQClientTactile, ZMQServerTactile
from real_ur5rg2.recording_time import (
    RecordingAlignmentError, RecordingPerformance, RecordingSampleUnavailable,
    RecordingTimeAlignment,
)


def eventually(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "Simulated worker did not reach the expected state"
        time.sleep(0.001)


def tactile_frame(captured=None):
    captured = time.time() if captured is None else captured
    return dict(force=np.ones((2, 2, 3), dtype=np.float32),
                force6d=np.zeros(6, dtype=np.float32), depth=np.zeros((2, 2)),
                rectify=np.zeros((2, 2, 3), dtype=np.uint8), valid=True,
                timestamp=captured + .001, host_timestamp=time.time(),
                capture_timestamp=captured)


def test_force_parser_handles_partial_packets_crc_faults_and_resynchronization():
    def packet(value):
        body = b'\x20\x4e' + struct.pack('<6h', value, -value, 300, 400, -500, 600)
        return body + struct.pack('<H', modbus_crc(body))
    parser = FTStreamParser()
    first, second = packet(100), packet(200)
    assert parser.feed(first[:7]) == []
    np.testing.assert_allclose(parser.feed(first[7:])[0], [1, -1, 3, .4, -.5, .6])
    damaged = bytearray(first)
    damaged[5] ^= 1
    frames = parser.feed(first + damaged + second[:8])
    assert frames == []  # A pre-fault frame must not be restamped as the newest.
    frames = parser.feed(second[8:])
    assert len(frames) == 1 and parser.crc_errors >= 1
    np.testing.assert_allclose(frames[0][:2], [2, -2])


def test_force_port_discovery_is_shared_by_polling_and_stream_modes():
    serial = SimpleNamespace(tools=SimpleNamespace(list_ports=SimpleNamespace(
        comports=lambda: [SimpleNamespace(device='/dev/ttyUSB-test', description='FTDI USB')]
    )))
    assert RobotiqFTS300ModbusSerial._find_serial_port(serial) == '/dev/ttyUSB-test'


def test_background_force_rejects_stale_values_and_owns_cleanup():
    gate = threading.Event()
    class Reader:
        timeout_s = .01
        closed = False
        calls = 0
        def read(self):
            self.calls += 1
            if self.calls > 1:
                gate.wait(1)
            return np.ones(6, dtype=np.float32)
        def cancel_read(self):
            gate.set()
        def close(self):
            self.closed = True
    device = Reader()
    reader = AsyncForceReader(device, max_age_s=.03)
    try:
        reader.wait_until_ready()
        value = reader.read()
        value[:] = 9
        assert reader.read()[0] == 1
        time.sleep(.04)
        with pytest.raises(RuntimeError, match='stale'):
            reader.read()
    finally:
        reader.close()
    assert device.closed and not reader._thread.is_alive()


def test_async_gripper_coalesces_targets_and_keeps_arm_idle_while_busy():
    gate, entered = threading.Event(), threading.Event()
    calls = []
    class Gripper:
        def move(self, width, force):
            calls.append(width)
            if len(calls) == 1:
                entered.set()
                gate.wait(1)
            return True
    worker = AsyncRG2Gripper(Gripper())
    try:
        worker.move(10, 4)
        assert entered.wait(1)
        worker.move(20, 4)
        worker.move(30, 4)
        robot = URRobot.__new__(URRobot)
        robot._closed, robot._gripper_worker = False, worker
        robot.command_lock = threading.Lock()
        robot.robot = SimpleNamespace(initPeriod=lambda: pytest.fail('Arm moved while RG2 was busy'))
        assert robot.command_tcp_pose(np.zeros(6)) is False
        assert robot.command_joint_state(np.zeros(7)) is False
        gate.set()
        eventually(lambda: not worker.status()['busy'])
        assert calls == [10, 30]
    finally:
        gate.set()
        worker.close()


def test_gripper_worker_errors_are_reported_instead_of_allowing_arm_commands():
    worker = AsyncRG2Gripper(SimpleNamespace(move=lambda *_: False))
    try:
        worker.move(0, 4)
        eventually(lambda: worker.status()['error'] is not None)
        with pytest.raises(RuntimeError, match='failed'):
            worker.move(110, 4)
    finally:
        worker.close()


def test_rtde_mapping_reuses_packet_time_and_rewarms_after_clock_reset():
    clock = RTDEClockMap()
    stamps = [clock.observe(10 + index * .008, 100 + index * .008) for index in range(50)]
    good = [stamp for stamp in stamps if stamp is not None]
    assert good and all(a.monotonic < b.monotonic for a, b in zip(good, good[1:]))
    assert clock.observe(good[-1].native, 101) is good[-1]
    epoch = clock.epoch
    assert clock.observe(1, 102) is None
    assert clock.epoch == epoch + 1


def test_xense_buffer_deduplicates_captures_and_does_not_refresh_stale_images():
    class Sensor:
        frame = tactile_frame()
        closed = False
        def read(self):
            return self.frame
        def close(self):
            self.closed = True
    sensor = Sensor()
    reader = AsyncXenseTactileReader(sensor, force_height=2, force_width=2,
                                    poll_interval_s=.001, max_age_s=.08)
    try:
        assert reader.wait_for_frame(1)
        first = sensor.frame['capture_timestamp']
        time.sleep(.002)
        sensor.frame = tactile_frame()
        second = sensor.frame['capture_timestamp']
        eventually(lambda: reader.read()['capture_timestamp'] == second)
        pair = reader.read_for_recording()
        assert pair['recording']['capture_timestamp'] == first
        assert pair['latest']['capture_timestamp'] == second
        assert reader.read_for_recording()['recording']['capture_timestamp'] == first
        assert reader.read_for_recording(second)['recording'] is None
        time.sleep(.09)
        with pytest.raises(RuntimeError, match='stopped updating|too old'):
            reader.read()
    finally:
        reader.close()
    assert sensor.closed


def test_sdk_lock_admits_waiting_sensor_threads_in_order():
    lock = _XenseSDKLock()
    order, threads = [], []
    lock.acquire()
    def run(index):
        with lock:
            order.append(index)
    try:
        for index in range(3):
            thread = threading.Thread(target=run, args=(index,))
            thread.start()
            threads.append(thread)
            eventually(lambda: len(lock._queue) == index + 1)
    finally:
        lock.release()
        for thread in threads:
            thread.join(1)
    assert order == [0, 1, 2]


@pytest.mark.parametrize('backend', [LeRobotRealSenseCamera, RealSenseCamera])
def test_camera_rgb_rpc_has_exact_frame_time_and_preserves_legacy_depth(backend):
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[..., 0] = 17
    color = SimpleNamespace(get_data=lambda: image, get_timestamp=lambda: 1234000.,
                            get_frame_timestamp_domain=lambda: 'timestamp_domain.global_time',
                            get_frame_number=lambda: 42)
    frames = SimpleNamespace(get_color_frame=lambda: color)
    camera = backend.__new__(backend)
    camera._device_id, camera._width, camera._height = 'simulated', 4, 3
    if backend is LeRobotRealSenseCamera:
        camera._camera = SimpleNamespace(is_connected=True,
            rs_pipeline=SimpleNamespace(try_wait_for_frames=lambda **_: (True, frames)),
            _postprocess_image=lambda array: array)
    else:
        camera._pipeline = SimpleNamespace(wait_for_frames=lambda **_: frames)
        camera._enable_depth, camera._flip = False, False
        camera._last_image = camera._last_depth = None
    rgb = camera.read_rgb_with_timestamp()
    assert 'depth' not in rgb and rgb['capture_timestamp'] == 1234
    assert rgb['frame_number'] == 42
    image[:] = 0
    assert np.any(rgb['image'])  # Native SDK memory must not mutate saved history.
    legacy_image, depth = camera.read()
    assert legacy_image.shape == (3, 4, 3) and depth.shape == (3, 4, 1)


class SimulatedDevice:
    def __init__(self):
        self.busy_reads = 0
        self.polls = 0
    def num_dofs(self):
        return 7
    def command_tcp_pose(self, **_):
        raise RuntimeError('simulated rejected command')
    def command_gripper(self, **_):
        self.busy_reads = 2
    def get_observations(self):
        self.polls += 1
        busy = self.busy_reads > 0
        self.busy_reads -= int(busy)
        return {'gripper_busy': busy, 'gripper_error': None}
    def read(self, *_):
        return np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2, 1))
    def read_rgb_with_timestamp(self, *_):
        return {'image': np.zeros((2, 2, 3), dtype=np.uint8), 'capture_timestamp': time.time()}


@pytest.mark.parametrize('kind', ['robot', 'camera', 'tactile'])
def test_zmq_timeout_recovery_does_not_leave_req_socket_in_invalid_state(kind):
    device = SimulatedDevice()
    server_type, client_type = {'robot': (ZMQServerRobot, ZMQClientRobot),
        'camera': (ZMQServerCamera, ZMQClientCamera),
        'tactile': (ZMQServerTactile, ZMQClientTactile)}[kind]
    server = server_type(device, port=0)
    port = int(server._socket.getsockopt(zmq.LAST_ENDPOINT).decode().rsplit(':', 1)[1])
    client = client_type(port=port, timeout_ms=150)
    request = client.num_dofs if kind == 'robot' else client.read
    thread = threading.Thread(target=server.serve)
    try:
        with pytest.raises(RuntimeError, match='ZMQ timeout'):
            request()
        thread.start()
        request()  # A second operation must succeed, with no EFSM or old reply.
        if kind == 'robot':
            with pytest.raises(RuntimeError, match='simulated rejected command'):
                client.command_tcp_pose(np.zeros(6))
            assert client.num_dofs() == 7
        elif kind == 'camera':
            assert 'depth' not in client.read_rgb_with_timestamp()
    finally:
        client.close()
        server.stop()
        if thread.ident is not None:
            thread.join(2)
            assert not thread.is_alive()
        server.close()


@pytest.mark.parametrize('blocking', [False, True])
def test_robot_client_keeps_inference_waits_and_collection_nonblocking(blocking):
    client = ZMQClientRobot.__new__(ZMQClientRobot)
    device = SimulatedDevice()
    client._blocking_gripper, client._timeout_ms = blocking, 1000
    def request(message):
        return getattr(device, message['method'])(**message.get('args', {}))
    client._request = request
    client.command_gripper(1)
    assert device.polls == (3 if blocking else 0)


def test_alignment_deduplicates_native_ur_packets_and_preserves_matching_tcp():
    alignment = RecordingTimeAlignment()
    now = time.time()
    observation = dict(observation_timestamp=now, rtde_timestamp=1.,
        rtde_timestamp_consistent=True, rtde_clock_epoch=0,
        joint_positions=np.zeros(7), tcp_pose=np.arange(6))
    alignment.add_state(observation)
    alignment.add_state({**observation, 'observation_timestamp': now - .000001})
    assert len(alignment.states.samples) == 1
    np.testing.assert_array_equal(alignment.states.samples[0].timing['tcp_pose'], np.arange(6))
    alignment.actions.add(now, 1.)
    for index in range(2):
        alignment.add_camera(index, dict(capture_timestamp=now, image=np.zeros((2, 2, 3)),
            timestamp_domain='timestamp_domain.global_time'))
    cameras, _, command = alignment.choose(now)
    assert command.value == 1.
    alignment.commit(cameras)
    with pytest.raises(RecordingSampleUnavailable):
        alignment.choose(now)
    with pytest.raises(RecordingAlignmentError, match='clock moved backwards'):
        alignment.add_state({**observation, 'rtde_timestamp': .9})


def test_preparation_statistics_do_not_count_discarded_warmup_as_recording():
    performance = RecordingPerformance(30, preparing=True)
    performance.start_loop(time.monotonic())
    performance.skips['waiting'] += 1
    performance.start_loop(time.monotonic())
    performance.mark_ready()
    performance.saved_rows += 1
    result = performance.summary()
    assert result['loops'] == 1 and result['saved_rows'] == 1
    assert result['skip_reasons'] == {}
    assert result['preparation']['waiting_loops'] == 1


@pytest.fixture(scope='module')
def collector():
    return importlib.import_module('real_ur5rg2.collect_lerobot_xenstactile_servo_control')


def test_png_writer_is_lossless_and_reports_encoder_failures(tmp_path):
    from real_ur5rg2.image_writer import RecordingImageWriter
    image = np.arange(60, dtype=np.uint8).reshape(5, 4, 3)
    writer = RecordingImageWriter(num_processes=1, num_threads=1)
    try:
        writer.save_image(image, tmp_path / 'rgb.png')
        writer.wait_until_done()
        np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / 'rgb.png')), image)
        writer.save_image(image, tmp_path / 'missing' / 'rgb.png')
        with pytest.raises(RuntimeError, match='Cannot write PNG'):
            writer.wait_until_done()
    finally:
        with pytest.raises(RuntimeError, match='Cannot write PNG'):
            writer.stop()


def test_resume_checks_layout_and_installs_background_writer(collector, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['collect', '--root', str(tmp_path), '--resume'])
    args = collector.parse_args()
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'data').mkdir()
    class Dataset:
        def __init__(self, *_args, **_kwargs):
            self.fps = 20
            self.features = {key: {'shape': (256, 256, 3)} for key in
                ('observation.image', 'observation.wrist_image')}
            self.image_writer = None
    monkeypatch.setattr(collector, 'LeRobotDataset', Dataset)
    monkeypatch.setattr(collector, 'RecordingImageWriter', lambda **_: 'background-writer')
    with pytest.raises(ValueError, match='20 fps'):
        collector.make_dataset(args)
    args.fps = 20
    with pytest.raises(ValueError, match='RGB shape'):
        collector.make_dataset(args)
    args.image_width = args.image_height = 256
    assert collector.make_dataset(args).image_writer == 'background-writer'


def test_auto_cycle_survives_alignment_skip_and_waits_for_gripper(collector, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['collect', '--num-episodes', '2', '--no-preview',
        '--no-collect-tactile', '--no-collect-force', '--image-width', '8', '--image-height', '6'])
    args = collector.parse_args()
    args.fps = 1000
    closed, saved, saved_keys = [], [], []
    class Robot:
        scalar = 0.
        busy = 0
        def __init__(self, **kwargs):
            assert kwargs['blocking_gripper'] is False
        def close(self):
            closed.append('robot')
        def num_dofs(self):
            return 7
        def get_tcp_pose(self):
            return collector.CALIBRATED_TCP_POSE_BASE.copy()
        def get_observations(self):
            busy = self.busy > 0
            self.busy -= int(busy)
            return dict(joint_positions=np.r_[np.zeros(6), self.scalar],
                tcp_pose=self.get_tcp_pose(), observation_timestamp=time.time(),
                gripper_busy=busy, gripper_error=None)
        def inverse_kinematics(self, *_args, **_kwargs):
            assert self.busy == 0
            return np.zeros(6)
        def command_joint_state(self, *_):
            assert self.busy == 0
        def command_gripper(self, scalar, **_):
            if scalar != self.scalar:
                self.scalar, self.busy = scalar, 3
    class Camera:
        def __init__(self, **_):
            pass
        def close(self):
            closed.append('camera')
        def read_rgb_with_timestamp(self, *_):
            return dict(image=np.zeros((6, 8, 3), dtype=np.uint8),
                capture_timestamp=time.time(), timestamp_domain='timestamp_domain.global_time')
    class Dataset:
        root = tmp_path
        episode_buffer = None
        meta = SimpleNamespace(total_episodes=0)
        features = {}
        def create_episode_buffer(self):
            return {}
        def clear_episode_buffer(self):
            self.episode_buffer = {}
        def stop_image_writer(self):
            closed.append('writer')
        def add_frame(self, frame, **_):
            assert robot.busy == 0
            saved_keys.append(set(frame))
            np.testing.assert_allclose(frame['observation.fingertip_center_pose_relative_to_nut'], 0, atol=1e-6)
        def save_episode(self):
            self.meta.total_episodes += 1
            saved.append(self.meta.total_episodes)
    class Keyboard:
        gripper_closed, screw_direction, loops = False, 0, 0
        def poll(self, _):
            self.loops += 1
            assert self.loops < 100, 'Automatic cycle stopped making progress'
            return np.zeros(3), np.eye(3), float(self.gripper_closed), False, (
                'random_reset_pose' if self.loops == 1 else None)
        def close(self):
            closed.append('keyboard')
    robot = Robot(blocking_gripper=False)
    def make_robot(**kwargs):
        assert kwargs['blocking_gripper'] is False
        return robot
    monkeypatch.setattr(collector, 'ZMQClientRobot', make_robot)
    monkeypatch.setattr(collector, 'ZMQClientCamera', Camera)
    monkeypatch.setattr(collector, 'make_dataset', lambda _: Dataset())
    monkeypatch.setattr(collector, 'KeyboardTeleop', Keyboard)
    monkeypatch.setattr(collector, 'sample_random_reset_tcp', lambda: np.zeros(6))
    monkeypatch.setattr(collector, 'AUTO_RESET_CLOSE_WAIT_SECONDS', 0)
    monkeypatch.setattr(collector, 'PreviewWindow', lambda **_: SimpleNamespace(
        enabled=False, show=lambda *_: None, close=lambda: closed.append('preview')))
    original_choose, attempts = collector.choose_recording_samples, []
    def choose(*values):
        attempts.append(1)
        if len(attempts) == 1:
            raise RecordingSampleUnavailable('simulated history warmup')
        return original_choose(*values)
    monkeypatch.setattr(collector, 'choose_recording_samples', choose)
    with ExitStack() as cleanup:
        collector._collect(args, cleanup)
    assert saved == [1, 2] and len(saved_keys) == 2
    assert all('observation.fingertip_center_pose_relative_to_nut' in keys for keys in saved_keys)
    assert closed.count('camera') == 2 and 'robot' in closed and 'writer' in closed
    report = json.loads((tmp_path / 'meta/recording_performance/episode_000000.json').read_text())
    assert report['saved_rows'] == 1 and report['preparation']['waiting_loops'] == 1


@pytest.mark.parametrize('backend', ['process', 'thread'])
def test_launcher_cleans_up_when_second_tactile_sensor_cannot_start(monkeypatch, backend):
    import launch_nodes as launch
    closed = []
    class Server:
        def __init__(self, *args, **kwargs):
            pass
        def serve(self):
            pytest.fail('A partially initialized hardware group must not serve')
        def stop(self):
            pass
        def close(self):
            closed.append('server')
    class Sensor:
        def __init__(self, serial, *, cam_id, force_height, force_width, name, **kwargs):
            if 'right' in name:
                raise RuntimeError('simulated second sensor startup failure')
        def close(self):
            closed.append('sensor')
    class AsyncSensor:
        def __init__(self, reader, *, max_age_s, **kwargs):
            assert max_age_s == .23
            self.reader = reader
        def wait_for_frame(self, _):
            return True
        def close(self):
            self.reader.close()
    monkeypatch.setattr(launch, 'make_robot', lambda _: SimpleNamespace(close=lambda: closed.append('robot')))
    monkeypatch.setattr(launch, 'ZMQServerRobot', Server)
    monkeypatch.setattr(launch, 'ZMQServerTactile', Server)
    monkeypatch.setattr(launch, 'XenseTactileReader', Sensor)
    monkeypatch.setattr(launch, 'ProcessXenseTactileReader', Sensor)
    monkeypatch.setattr(launch, 'AsyncXenseTactileReader', AsyncSensor)
    with pytest.raises(RuntimeError, match='second sensor startup failure'):
        launch.launch_robot_server(launch.Args(enable_camera=False, xense_max_age_s=.23,
                                                xense_backend=backend))
    assert closed.count('server') == 2
    assert closed.count('sensor') == 1 and closed[-1] == 'robot'
