"""Validate Xense SDK scheduling and ORT settings without connecting sensors."""
from queue import Empty, Queue
import sys
import threading
from types import SimpleNamespace

import pytest

from gello.tactile import xense


def test_sdk_lock_queues_requests_and_keeps_calls_exclusive(monkeypatch):
    lock = xense._XenseSDKLock()
    lock.acquire()
    main_holds_lock = True
    queued, entered = Queue(), Queue()
    release_first = threading.Event()
    original_wait = lock._condition.wait_for

    def wait_for(predicate):
        queued.put(threading.current_thread().name)
        return original_wait(predicate)

    monkeypatch.setattr(lock._condition, "wait_for", wait_for)
    errors = Queue()

    def worker():
        try:
            with lock:
                name = threading.current_thread().name
                entered.put(name)
                if name == "left":
                    assert release_first.wait(2.0), "SDK worker did not finish"
        except BaseException as error:
            errors.put(error)

    threads = []
    try:
        for name in ("left", "right", "left-again"):
            thread = threading.Thread(target=worker, name=name, daemon=True)
            threads.append(thread)
            thread.start()
            assert queued.get(timeout=2.0) == name
        lock.release()
        main_holds_lock = False
        assert entered.get(timeout=2.0) == "left"
        with pytest.raises(Empty):
            entered.get(timeout=.05)
        release_first.set()
        assert entered.get(timeout=2.0) == "right"
        assert entered.get(timeout=2.0) == "left-again"
    finally:
        release_first.set()
        if main_holds_lock:
            lock.release()
        for thread in threads:
            thread.join(timeout=2.0)
    assert all(not thread.is_alive() for thread in threads)
    assert errors.empty(), list(errors.queue)


def test_sdk_lock_context_releases_after_read_error():
    lock = xense._XenseSDKLock()
    with pytest.raises(ValueError, match="SDK read failed"):
        with lock:
            raise ValueError("SDK read failed")
    with lock:
        assert lock._held
    with pytest.raises(RuntimeError, match="unlocked"):
        lock.release()


@pytest.mark.parametrize("failure", [RuntimeError("wait failed"), KeyboardInterrupt()])
def test_sdk_lock_removes_interrupted_waiter(monkeypatch, failure):
    lock = xense._XenseSDKLock()
    lock.acquire()

    def interrupted(_):
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(lock._condition, "wait_for", interrupted)
        with pytest.raises(type(failure)):
            lock.acquire()
    lock.release()
    with lock:
        assert not lock._queue


@pytest.fixture
def ort(monkeypatch):
    class Options:
        def __init__(self):
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0
            self.config = {}

        def add_session_config_entry(self, key, value):
            self.config[key] = value

    class Session:
        def __init__(self, model, sess_options=None, *args, **kwargs):
            self.model = model
            self.options = sess_options
            self.extra_args = args
            self.extra_kwargs = kwargs
            if model == "broken":
                raise RuntimeError("model failed to load")

    runtime = SimpleNamespace(InferenceSession=Session, SessionOptions=Options)
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    return runtime


@pytest.mark.parametrize("existing_options", [False, True])
def test_xense_sessions_get_reference_thread_settings_and_restore_init(ort, existing_options, capsys):
    original_init = ort.InferenceSession.__init__
    options = ort.SessionOptions() if existing_options else None
    if options is not None:
        options.add_session_config_entry("keep_this_option", "yes")
    with xense._XENSE_SDK_LOCK, xense._xense_onnx_threads():
        session = ort.InferenceSession("sensor.onnx", options, providers=["CPUExecutionProvider"])
        assert session.options.intra_op_num_threads == 6
        assert session.options.inter_op_num_threads == 1
        assert session.options.config["session.intra_op.allow_spinning"] == "0"
        assert session.options.config["session.inter_op.allow_spinning"] == "0"
        assert session.extra_kwargs["providers"] == ["CPUExecutionProvider"]
        if existing_options:
            assert session.options is options
            assert options.config["keep_this_option"] == "yes"
    assert ort.InferenceSession.__init__ is original_init
    assert "intra_op_threads=6, inter_op_threads=1, spinning=off" in capsys.readouterr().out


def test_other_threads_keep_their_onnx_settings(ort):
    sessions = Queue()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    options.inter_op_num_threads = 2
    with xense._XENSE_SDK_LOCK, xense._xense_onnx_threads():
        thread = threading.Thread(
            target=lambda: sessions.put(ort.InferenceSession("other.onnx", options)), daemon=True)
        thread.start()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    session = sessions.get(timeout=2.0)
    assert session.options is options
    assert options.intra_op_num_threads == 3 and options.inter_op_num_threads == 2
    assert not options.config


def test_onnx_constructor_restored_when_model_initialization_fails(ort):
    original_init = ort.InferenceSession.__init__
    with pytest.raises(RuntimeError, match="model failed"):
        with xense._XENSE_SDK_LOCK, xense._xense_onnx_threads():
            ort.InferenceSession("broken")
    assert ort.InferenceSession.__init__ is original_init
    with xense._XENSE_SDK_LOCK:
        pass


@pytest.mark.parametrize("cam_id", [None, 3])
def test_sensor_creation_applies_thread_settings_in_sdk_lock(ort, monkeypatch, cam_id):
    original_init = ort.InferenceSession.__init__
    sessions, calls = [], []
    outputs = SimpleNamespace(TimeStamp=1, ForceResultant=2, Force=3, Depth=4,
                              Rectify=5, CaptureTimeStamp=6)

    def create(*args, **kwargs):
        assert xense._XENSE_SDK_LOCK._held
        calls.append((args, kwargs))
        session = ort.InferenceSession("sensor.onnx")
        assert session.options.intra_op_num_threads == 6
        assert session.options.inter_op_num_threads == 1
        sessions.append(session)
        return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "xensesdk", SimpleNamespace(
        Sensor=SimpleNamespace(OutputType=outputs, create=create)))
    xense.XenseTactileReader("fake" if cam_id is None else None, cam_id=cam_id)
    assert len(sessions) == 1
    expected = [(("fake",), {})] if cam_id is None else [((), {"cam_id": 3})]
    assert calls == expected
    assert ort.InferenceSession.__init__ is original_init
    with xense._XENSE_SDK_LOCK:
        pass


def test_failed_sensor_creation_restores_onnx_and_releases_sdk_lock(ort, monkeypatch):
    original_init = ort.InferenceSession.__init__

    def create(*_a, **_k):
        ort.InferenceSession("sensor.onnx")
        raise RuntimeError("device initialization failed")

    monkeypatch.setitem(sys.modules, "xensesdk", SimpleNamespace(Sensor=SimpleNamespace(
        OutputType=SimpleNamespace(CaptureTimeStamp=6), create=create)))
    with pytest.raises(RuntimeError, match="device initialization failed"):
        xense.XenseTactileReader("fake")
    assert ort.InferenceSession.__init__ is original_init
    with xense._XENSE_SDK_LOCK:
        pass
