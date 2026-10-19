"""FT300-S documented 100 Hz stream, with CRC and host timing provenance.

Reference: Robotiq FT300-S manual, section 4.3.2 (2021-08-31).
The device sends no timestamp or sequence number in this 16-byte protocol.
"""
import array
import math
import struct
import sys
import threading
import time

import numpy as np

from gello.force.robotiq_fts300 import ForceSample, RobotiqFTS300ModbusSerial


def modbus_crc(data: bytes) -> int:
    crc = 0xffff
    for value in data:
        crc ^= value
        for _ in range(8):
            crc = (crc >> 1) ^ 0xa001 if crc & 1 else crc >> 1
    return crc


class FTStreamParser:
    def __init__(self):
        self.buffer = bytearray()
        self.crc_errors = 0
        self.discarded_bytes = 0

    def feed(self, data: bytes) -> list[np.ndarray]:
        self.buffer.extend(data)
        frames = []
        while len(self.buffer) >= 16:
            position = self.buffer.find(b'\x20\x4e')
            if position < 0:
                self.discarded_bytes += len(self.buffer)-1
                del self.buffer[:-1]
                frames.clear()
                break
            if position:
                self.discarded_bytes += position
                del self.buffer[:position]
                frames.clear()
            if len(self.buffer) < 16:
                break
            packet = bytes(self.buffer[:16])
            if modbus_crc(packet[:14]) != struct.unpack('<H', packet[14:])[0]:
                self.crc_errors += 1
                self.discarded_bytes += 1
                del self.buffer[0]
                # Only return the consecutive valid suffix after a framing
                # fault. Earlier packets must not be restamped as latest data.
                frames.clear()
                continue
            del self.buffer[:16]
            raw = np.array(struct.unpack('<6h', packet[2:14]), dtype=np.float32)
            raw[:3] /= 100.0
            raw[3:] /= 1000.0
            frames.append(raw)
        return frames


class RobotiqFTS300Stream:
    frequency_hz = 100.0
    close_timeout_s = 2.0

    def __init__(self, port=None, *, slave_address=9, baudrate=19200,
                 timeout_s=.1, low_latency=True):
        import serial
        import serial.tools.list_ports

        if baudrate != 19200:
            raise ValueError("FT300-S stream requires the documented 19200 baud")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("FT stream timeout must be finite and positive")
        self.port = port or RobotiqFTS300ModbusSerial._find_serial_port(serial)
        if self.port is None:
            raise RuntimeError("No FT serial port found; use --force-serial-port")
        self.timeout_s = float(timeout_s)
        self.retries = 1
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._closed = False
        self._serial = None
        self._original_low_latency = None
        self.low_latency_active = False
        self.low_latency_error = None
        self.parser = FTStreamParser()
        self.frames_received = 0
        self._last_crc_warning = -math.inf
        self._reported_crc_errors = 0
        self._last_io = time.monotonic()
        self._last_sample = -math.inf
        self._stream_started = False
        try:
            self._serial = serial.Serial(self.port, baudrate, timeout=self.timeout_s,
                                         write_timeout=self.timeout_s, exclusive=True)
            if low_latency:
                self._enable_low_latency()
            self._stop_stream()
            payload = struct.pack('>BBHHBH', int(slave_address), 16, 410, 1, 2, 0x0200)
            self._stream_started = True
            self._serial.write(payload+struct.pack('<H', modbus_crc(payload)))
            self._serial.flush()
            self._last_io = time.monotonic()
        except Exception as exc:
            try:
                self.close()
            except Exception:
                pass
            raise RuntimeError(f"FT300-S stream startup failed: {exc}") from exc
        print(f"Robotiq FT300-S stream port={self.port}, output=100 Hz, "
              f"baudrate=19200, low_latency={self.low_latency_active}")

    def _enable_low_latency(self):
        try:
            import fcntl
            import termios
            buffer = array.array('i', [0]*32)
            fcntl.ioctl(self._serial.fileno(), termios.TIOCGSERIAL, buffer)
            self._original_low_latency = bool(buffer[4] & 0x2000)
            self._serial.set_low_latency_mode(True)
            self.low_latency_active = True
        except (ImportError, AttributeError, OSError, ValueError) as exc:
            self.low_latency_error = str(exc)
            print(f"FT stream: adapter low-latency mode unavailable: {exc}")

    def _stop_stream(self):
        # Continuous bursts are required; isolated FF bytes can collide with
        # outgoing frames without actually stopping this hardware's stream.
        start = time.monotonic()
        while time.monotonic()-start < .55:
            self._serial.write(b'\xff'*50)
            self._serial.flush()
            # USB tcdrain may finish before the UART has transmitted its FIFO.
            # Pace each 26 ms burst so the start command follows the final FF.
            time.sleep(50*10/19200+.004)
        self._serial.reset_input_buffer()

    def read_samples(self) -> list[ForceSample]:
        with self._lock:
            if self._closed or self._cancel.is_set():
                raise RuntimeError("FT stream is closed or cancelled")
            entered = time.monotonic()
            bound = min(entered, self._last_io)
            deadline = entered+self.timeout_s
            while time.monotonic() < deadline:
                if self._cancel.is_set():
                    raise RuntimeError("FT stream read cancelled")
                data = self._serial.read(max(1, min(4096, self._serial.in_waiting)))
                remaining = self._serial.in_waiting
                if remaining:
                    data += self._serial.read(min(4096, remaining))
                received = time.monotonic()
                if not data:
                    continue
                frames = self.parser.feed(data)
                if not frames:
                    continue
                self._last_io = received
                samples = []
                wire_time = 16*10/19200
                for index, wrench in enumerate(frames):
                    self.frames_received += 1
                    # Complete-packet receipt is real. Earlier times within a
                    # USB batch use the documented cadence, not a device clock.
                    estimated = received-wire_time-(len(frames)-1-index)/self.frequency_hz
                    if estimated <= self._last_sample:
                        # The overlap is timing ambiguity, not permission to
                        # restamp an older frame as a newly sampled force.
                        continue
                    self._last_sample = estimated
                    samples.append(ForceSample(
                        wrench, min(bound, estimated), received,
                        sample_monotonic=estimated,
                        timing_basis="stream_frame_start_estimate",
                        sequence=self.frames_received,
                        timing_note="Host receipt minus 16-byte UART duration; USB batches use nominal 100 Hz spacing. Device sample/filter delay is unknown; sequence is a host receive counter."))
                if samples:
                    if (self.parser.crc_errors > self._reported_crc_errors and
                            received-self._last_crc_warning >= 1.0):
                        print("FT300-S stream recovered CRC-valid frames; "
                              f"discarded CRC candidates={self.parser.crc_errors}, "
                              f"discarded bytes={self.parser.discarded_bytes}",
                              file=sys.stderr)
                        self._last_crc_warning = received
                        self._reported_crc_errors = self.parser.crc_errors
                    return samples
            raise RuntimeError(
                "FT stream did not deliver a CRC-valid frame before timeout "
                f"({self.timeout_s:.3f}s; CRC errors={self.parser.crc_errors}, "
                f"discarded bytes={self.parser.discarded_bytes})")

    def read(self) -> np.ndarray:
        return self.read_samples()[-1].wrench.copy()

    def cancel_read(self):
        self._cancel.set()
        if self._serial is not None and self._serial.is_open:
            self._serial.cancel_read()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            port = self._serial
            if port is None:
                return
            try:
                if port.is_open and self._stream_started:
                    self._stop_stream()
            finally:
                try:
                    if port.is_open and self._original_low_latency is not None:
                        port.set_low_latency_mode(self._original_low_latency)
                finally:
                    port.close()
