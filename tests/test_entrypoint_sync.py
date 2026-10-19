"""Exercise collection/inference entry points with simulated I/O only."""
from contextlib import ExitStack
import importlib
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from real_ur5rg2.recording_time import TimedSample


MODULES = [
    'real_ur5rg2.collect_lerobot_smolvla_real_xenstactile_w_force',
    'tactile_policy.infer_tactile_servo_nutbolt_miben_lhy',
    'tactile_policy.infer_tactile_servo_nutbolt_miben_lhy_new',
]


@pytest.fixture(params=MODULES)
def entry(request):
    return importlib.import_module(request.param)


def arguments(module, monkeypatch, *extra):
    argv = ['entry']
    if module.__name__.startswith('tactile_policy.'):
        argv += ['--checkpoint', '/tmp/not-loaded-by-regression.pt']
    monkeypatch.setattr(sys, 'argv', argv + list(extra))
    return module.parse_args()


@pytest.mark.parametrize('source', ['fts300-stream', 'modbus-serial'])
def test_serial_values_are_cached_off_the_control_thread_and_expire(entry, monkeypatch, source):
    args = arguments(entry, monkeypatch, '--force-source', source, '--force-max-age-s', '.03')
    gate = threading.Event()
    class Serial:
        calls = 0
        timeout_s = .1
        closed = 0
        callers = []
        def read(self):
            self.callers.append(threading.get_ident())
            self.calls += 1
            if self.calls > 1:
                gate.wait(1)
            return np.ones(6, dtype=np.float32)
        def cancel_read(self):
            gate.set()
        def close(self):
            self.closed += 1
    serial = Serial()
    monkeypatch.setattr(entry, 'RobotiqFTS300Stream', lambda **_: serial)
    monkeypatch.setattr(entry, 'RobotiqFTS300ModbusSerial', lambda **_: serial)
    robot = SimpleNamespace(get_tcp_force=lambda: pytest.fail('Serial worker touched robot RPC'))
    reader = entry.ForceReader(args, robot)
    try:
        # Legacy collection starts its polling worker after tare; inference and
        # continuous streaming start at construction, before waiting for input.
        start = getattr(reader, 'start_background', None)
        if callable(start):
            start()
        assert reader._background is not None
        value = reader.read()
        value[0] = 99
        assert reader.read()[0] == 1
        assert all(caller != threading.get_ident() for caller in serial.callers)
        time.sleep(.04)
        with pytest.raises(RuntimeError, match='stale'):
            reader.read()
    finally:
        reader.close()
    reader.close()
    assert serial.closed == 1


def test_stream_startup_failure_closes_its_background_reader(entry, monkeypatch):
    args = arguments(entry, monkeypatch)
    serial = SimpleNamespace()
    closed = []
    class Background:
        def __init__(self, backend, **_):
            assert backend is serial
        def wait_until_ready(self):
            raise RuntimeError('simulated serial startup failure')
        def close(self):
            closed.append('serial')
    monkeypatch.setattr(entry, 'RobotiqFTS300Stream', lambda **_: serial)
    monkeypatch.setattr(entry, 'AsyncForceReader', Background)
    with pytest.raises(RuntimeError, match='serial startup failure'):
        entry.ForceReader(args, object())
    assert closed == ['serial']


@pytest.mark.parametrize('source', ['ur-rtde', 'modbus-tcp'])
def test_existing_nonserial_sources_do_not_use_the_serial_worker(entry, monkeypatch, source):
    args = arguments(entry, monkeypatch, '--force-source', source)
    closed = []
    values = np.arange(6, dtype=np.float32)
    monkeypatch.setattr(entry, 'AsyncForceReader', lambda *_args, **_kwargs:
                        pytest.fail('Nonserial source was put on a serial worker'))
    monkeypatch.setattr(entry, 'RobotiqFTS300ModbusTCP', lambda **_: SimpleNamespace(
        read=lambda: values, close=lambda: closed.append('tcp')))
    reader = entry.ForceReader(args, SimpleNamespace(get_tcp_force=lambda: values))
    try:
        np.testing.assert_array_equal(reader.read(), values)
    finally:
        reader.close()
    assert closed == (['tcp'] if source == 'modbus-tcp' else [])


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_invalid_force_freshness_is_rejected_before_opening_devices(entry, monkeypatch, value):
    with pytest.raises(SystemExit) as error:
        arguments(entry, monkeypatch, '--force-max-age-s', value)
    assert error.value.code == 2


@pytest.mark.parametrize('name', MODULES[1:])
def test_inference_takes_both_tactile_snapshots_before_formatting(monkeypatch, name):
    module = importlib.import_module(name)
    order = []
    source = module.TactileSource.__new__(module.TactileSource)
    source.args = SimpleNamespace(xense_force_height=2, xense_force_width=2)
    def read(side):
        order.append('read-' + side)
        return side
    source.left = SimpleNamespace(read=lambda: read('left'))
    source.right = SimpleNamespace(read=lambda: read('right'))
    def format_frame(frame, *_):
        order.append('format-' + frame)
        return {'valid': True}
    monkeypatch.setattr(module, 'format_tactile_frame', format_frame)
    source.read()
    assert order == ['read-left', 'read-right', 'format-left', 'format-right']


@pytest.mark.parametrize('name', MODULES[1:])
def test_inference_closes_all_io_even_if_serial_cleanup_fails(monkeypatch, name):
    module = importlib.import_module(name)
    args = arguments(module, monkeypatch, '--grasp-placement-test', '--use-force-safety')
    closed = []
    robot = SimpleNamespace(num_dofs=lambda: 7, get_tcp_pose=lambda: np.zeros(6),
                            close=lambda: closed.append('robot'))
    frame = dict(force=np.zeros((2, 2, 3)), force6d=np.zeros(6), valid=True)
    tactile = SimpleNamespace(read=lambda: (frame, frame), close=lambda: closed.append('tactile'))
    def close_force():
        closed.append('force')
        raise RuntimeError('simulated serial close failure')
    monkeypatch.setattr(module, 'parse_args', lambda: args)
    monkeypatch.setattr(module, 'resolve_device', lambda _: 'cpu')
    monkeypatch.setattr(module, 'load_model', lambda *_: (None, None))
    monkeypatch.setattr(module, 'ZMQClientRobot', lambda **_: robot)
    monkeypatch.setattr(module, 'TactileSource', lambda *_: tactile)
    monkeypatch.setattr(module, 'ForceReader', lambda *_: SimpleNamespace(
        read=lambda: np.zeros(6), close=close_force))
    monkeypatch.setattr(module, 'calibrate_force_bias', lambda *_: np.zeros(6))
    if hasattr(module, 'TactileForceLogger'):
        monkeypatch.setattr(module, 'TactileForceLogger', lambda *_: SimpleNamespace(
            close=lambda: closed.append('logger')))
    with pytest.raises(RuntimeError, match='serial close failure'):
        module.main()
    assert closed[-3:] == ['force', 'tactile', 'robot']


def test_additional_collector_resumes_with_background_png_and_keeps_force_fields(monkeypatch, tmp_path):
    module = importlib.import_module(MODULES[0])
    args = arguments(module, monkeypatch, '--root', str(tmp_path / 'new'))
    class Dataset:
        fps = 20
        features = {key: {'shape': (256, 256, 3)} for key in
                    ('observation.image', 'observation.wrist_image')}
        def __init__(self, *_args, **_kwargs):
            pass
        @classmethod
        def create(cls, **kwargs):
            assert kwargs['image_writer_threads'] == kwargs['image_writer_processes'] == 0
            assert 'observation.tactile_left.force' in kwargs['features']
            assert 'observation.tactile_left.distributed_force' not in kwargs['features']
            assert 'observation.fingertip_center_pose_relative_to_nut' not in kwargs['features']
            return cls()
    monkeypatch.setattr(module, 'LeRobotDataset', Dataset)
    monkeypatch.setattr(module, 'RecordingImageWriter', lambda **_: 'background PNG')
    assert module.make_dataset(args).image_writer == 'background PNG'
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'data').mkdir()
    args.root, args.resume = str(tmp_path), True
    with pytest.raises(ValueError, match='20 fps'):
        module.make_dataset(args)
    args.fps = 20
    args.image_width = args.image_height = 256
    assert module.make_dataset(args).image_writer == 'background PNG'
    args.resume = False
    monkeypatch.setattr('builtins.input', lambda _: 'c')
    assert module.make_dataset(args).image_writer == 'background PNG'


def test_additional_collector_records_aligned_rgb_tactile_and_cached_force(monkeypatch, tmp_path):
    module = importlib.import_module(MODULES[0])
    args = arguments(module, monkeypatch, '--num-episodes', '1', '--no-preview',
                     '--image-width', '8', '--image-height', '6')
    args.fps = 1000
    closed, rows, events = [], [], []
    class Dataset:
        root, episode_buffer = tmp_path, None
        meta = SimpleNamespace(total_episodes=0)
        features = {key: {} for key in [*module.TACTILE_DATASET_KEYS, module.FORCE_TORQUE_KEY]}
        features.update({key: {'dtype': 'image', 'shape': (
            args.xense_depth_height, args.xense_depth_width, 3)} for key in (
                'observation.tactile_left.depth', 'observation.tactile_right.depth')})
        def create_episode_buffer(self):
            return {}
        def add_frame(self, frame, **_):
            rows.append(frame)
        def save_episode(self):
            self.meta.total_episodes += 1
        def stop_image_writer(self):
            closed.append('writer')
    dataset = Dataset()
    class Robot:
        def __init__(self, **kwargs):
            assert kwargs['blocking_gripper'] is False
        def num_dofs(self):
            return 7
        def get_tcp_pose(self):
            return np.zeros(6)
        def get_observations(self):
            return dict(joint_positions=np.zeros(7), observation_timestamp=time.time())
        def close(self):
            closed.append('robot')
    class Camera:
        def __init__(self, **_):
            pass
        def read_rgb_with_timestamp(self, *_):
            return dict(image=np.zeros((6, 8, 3), dtype=np.uint8), capture_timestamp=time.time(),
                        timestamp_domain='timestamp_domain.global_time')
        def close(self):
            closed.append('camera')
    class Tactile:
        def __init__(self, **_):
            pass
        def read(self):
            return dict(force=np.zeros((2, 2, 3)), force6d=np.zeros(6),
                depth=np.zeros((2, 2)), rectify=np.zeros((2, 2, 3), dtype=np.uint8),
                valid=True, timestamp=time.time(), capture_timestamp=time.time())
        def read_for_recording(self, *_args, **_kwargs):
            frame = self.read()
            return {'latest': frame, 'recording': frame}
        def close(self):
            closed.append('tactile')
    class Force:
        crc_errors = 0
        def __init__(self, *_):
            pass
        def start_background(self):
            events.append('force-background')
        def read(self):
            return np.zeros(6)
        def read_for_recording(self, target, _):
            assert 'force-background' in events
            return TimedSample(target, np.zeros(6), {'basis': 'simulated cached sample'})
        def close(self):
            closed.append('force')
    class Keyboard:
        gripper_closed, screw_direction = False, 0
        def poll(self, _):
            events.append('poll')
            assert len(events) < 100, 'Simulated collection never became ready'
            event = 'done' if rows else 'start' if events.count('poll') == 1 else None
            return np.zeros(3), np.eye(3), 0., False, event
        def close(self):
            closed.append('keyboard')
    monkeypatch.setattr(module, 'ZMQClientRobot', Robot)
    monkeypatch.setattr(module, 'ZMQClientCamera', Camera)
    monkeypatch.setattr(module, 'ZMQClientTactile', Tactile)
    monkeypatch.setattr(module, 'ForceReader', Force)
    monkeypatch.setattr(module, 'make_dataset', lambda _: dataset)
    monkeypatch.setattr(module, 'KeyboardTeleop', Keyboard)
    monkeypatch.setattr(module, 'calibrate_force_bias', lambda *_: np.zeros(6))
    monkeypatch.setattr(module, 'PreviewWindow', lambda **_: SimpleNamespace(
        enabled=False, show=lambda *_: None, close=lambda: closed.append('preview')))
    with ExitStack() as cleanup:
        module._collect(args, cleanup)
    assert len(rows) == 1 and dataset.meta.total_episodes == 1
    assert 'observation.tactile_left.force' in rows[0]
    assert 'observation.force_torque' in rows[0]
    assert closed.count('camera') == closed.count('tactile') == 2
    assert 'force' in closed and 'writer' in closed and 'robot' in closed
    assert (tmp_path / 'meta/sensor_timing/episode_000000.json').is_file()
