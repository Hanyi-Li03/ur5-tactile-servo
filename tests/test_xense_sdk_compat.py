"""Xense SDK compatibility regressions; no physical devices are created."""
from contextlib import contextmanager, nullcontext
from enum import Enum
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from gello.tactile import xense


def install_sdk(monkeypatch, *, native_capture=False, output_type=None):
    names = ['TimeStamp', 'ForceResultant', 'Force', 'Depth', 'Rectify']
    if native_capture:
        names.append('CaptureTimeStamp')
    output_type = output_type or Enum('OutputType', {name: i for i, name in enumerate(names)})
    class Device:
        released = 0
        timestamp = time.time()
        capture = timestamp - .01
        image = np.ones((2, 3, 3), dtype=np.uint8)
        requested = None
        short = False
        def selectSensorInfo(self, *outputs):
            self.requested = outputs
            if self.short:
                return ()
            values = {
                'TimeStamp': self.timestamp,
                'ForceResultant': np.arange(6, dtype=np.float32),
                'Force': np.ones((2, 3, 3), dtype=np.float32),
                'Depth': np.ones((2, 3), dtype=np.float32),
                'Rectify': self.image,
                'CaptureTimeStamp': self.capture,
            }
            return tuple(values[item.name] for item in outputs)
        def release(self):
            self.released += 1
    device = Device()
    creates = []
    def create(*args, **kwargs):
        creates.append((args, kwargs))
        return device
    monkeypatch.setitem(sys.modules, 'xensesdk', SimpleNamespace(
        Sensor=SimpleNamespace(OutputType=output_type, create=create)))
    monkeypatch.setattr(xense, '_xense_onnx_threads', nullcontext)
    return device, creates


def make_reader(**kwargs):
    return xense.XenseTactileReader('simulated', force_height=2, force_width=3, **kwargs)


def test_old_sdk_initializes_without_capture_enum_and_returns_all_payloads(monkeypatch, capsys):
    device, creates = install_sdk(monkeypatch)
    reader = make_reader()
    try:
        frame = reader.read()
        assert len(creates) == 1 and len(device.requested) == 5
        assert frame['valid'] and frame['capture_timestamp'] == device.timestamp
        np.testing.assert_array_equal(frame['force6d'], np.arange(6))
        assert frame['force'].shape == (2, 3, 3)
        assert frame['depth'].shape == (2, 3)
        assert 'estimated software frame time' in capsys.readouterr().out
        device.image[:] = 7
        assert np.all(frame['rectify'] == 1)  # SDK-owned buffers are snapshotted.
    finally:
        reader.close()
        reader.close()
    assert device.released == 1
    with pytest.raises(RuntimeError, match='closed'):
        reader.read()


def test_real_installed_sdk_enum_is_supported_without_opening_any_device(monkeypatch):
    sdk = pytest.importorskip('xensesdk')
    output_type = sdk.Sensor.OutputType
    device, _ = install_sdk(monkeypatch, output_type=output_type)
    reader = make_reader()
    try:
        assert reader.read()['valid']
        assert all(item in output_type for item in device.requested)
    finally:
        reader.close()


def test_native_capture_timestamp_is_not_replaced_or_image_deduplicated(monkeypatch):
    device, _ = install_sdk(monkeypatch, native_capture=True)
    reader = make_reader()
    try:
        first = reader.read()
        device.capture += .001
        second = reader.read()
        assert len(device.requested) == 6
        assert first['capture_timestamp'] != second['capture_timestamp']
        assert second['capture_timestamp'] == device.capture
        device.capture = None
        assert not reader.read()['valid']
    finally:
        reader.close()


def test_old_sdk_reinference_does_not_restamp_identical_images(monkeypatch):
    device, _ = install_sdk(monkeypatch)
    reader = make_reader()
    try:
        first = reader.read()['capture_timestamp']
        device.timestamp += .005
        assert reader.read()['capture_timestamp'] == first
        device.image = device.image + 1
        assert reader.read()['capture_timestamp'] == device.timestamp
        last = reader.read()['capture_timestamp']
        device.image = device.image + 1
        assert reader.read()['capture_timestamp'] == last  # Must also advance SDK time.
        device.timestamp += .005
        assert reader.read()['capture_timestamp'] == device.timestamp
    finally:
        reader.close()


@pytest.mark.parametrize('stamp', [None, 0., -1., float('nan'), float('inf')])
def test_old_sdk_invalid_timestamp_fails_closed(monkeypatch, stamp):
    device, _ = install_sdk(monkeypatch)
    device.timestamp = stamp
    reader = make_reader()
    try:
        with pytest.raises(RuntimeError, match='invalid SDK timestamp'):
            reader.read()
    finally:
        reader.close()


@pytest.mark.parametrize('image', [None, np.empty((0, 0)), np.array(1), np.array([[object()]])])
def test_old_sdk_missing_image_does_not_bypass_deduplication(monkeypatch, image):
    device, _ = install_sdk(monkeypatch)
    device.image = image
    reader = make_reader()
    try:
        with pytest.raises(RuntimeError, match='valid Rectify image'):
            reader.read()
    finally:
        reader.close()


def test_old_sdk_reversed_time_fails_closed(monkeypatch):
    device, _ = install_sdk(monkeypatch)
    reader = make_reader()
    try:
        reader.read()
        device.timestamp -= 1
        with pytest.raises(RuntimeError, match='went backwards'):
            reader.read()
    finally:
        reader.close()


def test_short_sdk_payload_is_invalid_instead_of_becoming_a_new_frame(monkeypatch):
    device, _ = install_sdk(monkeypatch)
    device.short = True
    reader = make_reader()
    try:
        assert not reader.read()['valid']
    finally:
        reader.close()


def test_initialization_error_releases_sensor(monkeypatch):
    device, _ = install_sdk(monkeypatch)
    @contextmanager
    def broken_context():
        yield
        raise RuntimeError('simulated initialization failure after opening')
    monkeypatch.setattr(xense, '_xense_onnx_threads', broken_context)
    with pytest.raises(RuntimeError, match='simulated initialization failure'):
        make_reader()
    assert device.released == 1


def test_unsupported_required_enum_fails_before_opening_sensor(monkeypatch):
    _, creates = install_sdk(monkeypatch, output_type=Enum('Incomplete', {'TimeStamp': 1}))
    with pytest.raises(AttributeError):
        make_reader()
    assert not creates


def test_old_sdk_duplicate_inference_eventually_expires(monkeypatch):
    device, _ = install_sdk(monkeypatch)
    reader = make_reader()
    async_reader = xense.AsyncXenseTactileReader(
        reader, force_height=2, force_width=3, poll_interval_s=.001, max_age_s=.5)
    try:
        assert async_reader.wait_for_frame(1)
        initial = async_reader.read()['capture_timestamp']
        device.timestamp += .01
        time.sleep(.02)
        assert async_reader.read()['capture_timestamp'] == initial
        with async_reader._lock:
            async_reader._last_progress = time.monotonic() - 1
        with pytest.raises(RuntimeError, match='stopped updating'):
            async_reader.read()
    finally:
        async_reader.close()
    assert device.released == 1


def test_launcher_releases_reader_if_async_wrapper_cannot_start(monkeypatch):
    import launch_nodes as launch
    closed = []
    class Server:
        def __init__(self, *args, **kwargs):
            pass
        def serve(self):
            pytest.fail('No serving after partial initialization')
        def stop(self):
            pass
        def close(self):
            closed.append('server')
    def failed_worker(*args, **kwargs):
        raise RuntimeError('simulated worker startup failure')
    monkeypatch.setattr(launch, 'make_robot', lambda _: SimpleNamespace(
        close=lambda: closed.append('robot')))
    monkeypatch.setattr(launch, 'ZMQServerRobot', Server)
    monkeypatch.setattr(launch, 'XenseTactileReader', lambda *args, **kwargs: SimpleNamespace(
        close=lambda: closed.append('sensor')))
    monkeypatch.setattr(launch, 'ProcessXenseTactileReader', lambda *args, **kwargs: SimpleNamespace(
        close=lambda: closed.append('sensor')))
    monkeypatch.setattr(launch, 'AsyncXenseTactileReader', failed_worker)
    with pytest.raises(RuntimeError, match='simulated worker startup failure'):
        launch.main(launch.Args(enable_camera=False))
    assert closed == ['sensor', 'server', 'robot']
