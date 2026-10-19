"""Exercise real spawned workers and large shared frames without hardware."""

from functools import partial
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import pytest

from gello.tactile.xense import AsyncXenseTactileReader, _XENSE_SDK_LOCK
from gello.tactile.xense_process import ProcessXenseTactileReader


class SimulatedReader:
    def __init__(self, serial_number=None, *, name, cam_id, force_height,
                 force_width, close_marker=None, rendezvous=None, fault=None):
        if serial_number == 'startup-error':
            raise RuntimeError('simulated device initialization failure')
        if serial_number == 'startup-crash':
            os._exit(7)
        if serial_number == 'startup-blocked':
            time.sleep(60)
        self.serial = serial_number
        self.close_marker = close_marker
        self.rendezvous = rendezvous
        self.fault = fault
        self.counter = 0
        self.capture = time.time()

    def read(self):
        with _XENSE_SDK_LOCK:
            if self.counter == 0 and self.rendezvous is not None:
                self.rendezvous.wait(timeout=5)
            if self.serial == 'blocked':
                time.sleep(60)
            if self.fault is not None and self.fault.is_set():
                raise RuntimeError('simulated SDK read failure')
            time.sleep(.01)
            self.counter += 1
            if self.serial != 'frozen':
                self.capture = time.time() - .005
            return {
                'valid': True,
                'force': np.full((2, 3, 3), self.counter, dtype=np.float32),
                'force6d': np.full(6, self.counter, dtype=np.float32),
                'rectify': np.full((700, 400, 3), self.counter % 256, dtype=np.uint8),
                'depth': np.full((700, 400), self.counter, dtype=np.float32),
                'timestamp': self.capture,
                'capture_timestamp': self.capture,
                'host_timestamp': time.time(),
                'worker_pid': os.getpid(),
            }

    def close(self):
        if self.close_marker is not None:
            Path(self.close_marker).write_text(str(os.getpid()))


def make_reader(serial='live', **kwargs):
    kwargs.setdefault('_reader_factory', SimulatedReader)
    return ProcessXenseTactileReader(serial, force_height=2, force_width=3, **kwargs)


def test_two_sensors_run_in_separate_sdk_processes():
    # Both workers must enter their SDK lock together to pass this barrier.
    # A shared global SDK lock would serialize them and fail the rendezvous.
    barrier = mp.get_context('spawn').Barrier(2)
    factory = partial(SimulatedReader, rendezvous=barrier)
    left = make_reader('left', _reader_factory=factory)
    try:
        right = make_reader('right', _reader_factory=factory)
        try:
            frames = [left.read(), right.read()]
            pids = [frame['worker_pid'] for frame in frames]
            assert len(set(pids)) == 2 and os.getpid() not in pids
            assert all(frame['valid'] for frame in frames)
        finally:
            right.close()
    finally:
        left.close()


def test_shared_memory_keeps_only_latest_large_frame_and_preserves_snapshot(tmp_path):
    marker = tmp_path / 'device_closed'
    reader = make_reader(_reader_factory=partial(SimulatedReader, close_marker=str(marker)))
    try:
        first = reader.read()
        initial = int(first['force6d'][0])
        time.sleep(.15)
        latest = reader.read()
        counter = int(latest['force6d'][0])
        assert counter > initial + 3
        assert time.time() - latest['capture_timestamp'] < .1
        assert np.all(latest['depth'] == counter)
        assert np.all(latest['rectify'] == counter % 256)
        assert np.all(first['rectify'] == initial % 256)
        assert np.all(first['depth'] == initial)
    finally:
        process = reader._process
        reader.close()
        reader.close()
    assert not process.is_alive() and marker.read_text() == str(process.pid)
    with pytest.raises(RuntimeError, match='closed'):
        reader.read()


def test_startup_failure_is_reported_and_child_is_reaped():
    before = {child.pid for child in mp.active_children()}
    with pytest.raises(RuntimeError, match='simulated device initialization failure'):
        make_reader('startup-error')
    assert {child.pid for child in mp.active_children()} == before


@pytest.mark.parametrize('serial, message, kwargs', [
    ('startup-crash', 'exited during startup', {}),
    ('startup-blocked', 'startup timed out', {'startup_timeout_s': .2}),
])
def test_native_startup_crash_or_hang_does_not_leak_child(serial, message, kwargs):
    before = {child.pid for child in mp.active_children()}
    with pytest.raises(RuntimeError, match=message):
        make_reader(serial, **kwargs)
    assert {child.pid for child in mp.active_children()} == before


def test_sdk_error_reaches_parent_and_a_new_result_recovers():
    fault = mp.get_context('spawn').Event()
    reader = make_reader(_reader_factory=partial(SimulatedReader, fault=fault))
    try:
        initial = reader.read()['capture_timestamp']
        fault.set()
        deadline = time.monotonic() + 2
        while True:
            try:
                reader.read()
            except RuntimeError as exc:
                assert 'simulated SDK read failure' in str(exc)
                break
            assert time.monotonic() < deadline
        fault.clear()
        while True:
            try:
                assert reader.read()['capture_timestamp'] > initial
                break
            except RuntimeError:
                assert time.monotonic() < deadline
    finally:
        reader.close()


def test_worker_crash_does_not_return_old_shared_frame_or_hang():
    reader = make_reader()
    try:
        reader.read()
        reader._process.terminate()
        reader._process.join(2)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match='SDK process exited'):
            reader.read()
        assert time.monotonic() - started < .2
    finally:
        reader.close()


def test_shared_frame_lock_wait_is_bounded():
    reader = make_reader()
    try:
        reader.read()
        with reader._lock:
            reader._available.set()
            started = time.monotonic()
            with pytest.raises(RuntimeError, match='shared frame is unavailable'):
                reader.read()
            assert time.monotonic() - started < .3
        assert reader.read()['valid']
    finally:
        reader.close()


def test_repeated_capture_still_expires_through_process_backend():
    process_reader = make_reader('frozen')
    reader = AsyncXenseTactileReader(process_reader, max_age_s=.15,
                                    force_height=2, force_width=3)
    try:
        assert reader.wait_for_frame(2)
        captured = reader.read()['capture_timestamp']
        time.sleep(.18)
        with pytest.raises(RuntimeError, match='stopped updating|too old'):
            reader.read()
        assert reader._capture_key == captured
    finally:
        reader.close()


def test_close_terminates_a_worker_stuck_in_native_read():
    reader = make_reader('blocked')
    process = reader._process
    started = time.monotonic()
    reader.close()
    assert not process.is_alive()
    assert time.monotonic() - started < 4


@pytest.mark.parametrize('value', [-.1, float('nan'), float('inf')])
def test_invalid_poll_interval_is_rejected_before_starting_child(value):
    with pytest.raises(ValueError, match='poll_interval_s'):
        make_reader(poll_interval_s=value)
