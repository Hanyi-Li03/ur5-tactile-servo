import socket
import struct
import threading
import time
from dataclasses import dataclass, replace
from collections import deque
from typing import Literal

import numpy as np


ModbusFunction = Literal["holding", "input"]
ModbusValueFormat = Literal["int16", "int32", "float32"]
ModbusWordOrder = Literal["big", "little"]


@dataclass(frozen=True)
class ForceSample:
    wrench: np.ndarray
    read_started_monotonic: float
    received_monotonic: float
    sample_monotonic: float | None = None
    timing_basis: str = "serial_query_midpoint"
    sequence: int | None = None
    timing_note: str | None = None

    @property
    def timestamp_monotonic(self):
        return ((self.read_started_monotonic+self.received_monotonic)/2
                if self.sample_monotonic is None else self.sample_monotonic)


class AsyncForceReader:
    """Own one force reader and publish its latest sample without blocking.

    Freshness is measured from the start of the read, so a slow transaction
    cannot make an old result appear new just because it has just returned.
    The worker owns the underlying reader until close() has joined it.
    """

    def __init__(self, reader, *, max_age_s: float = 0.1) -> None:
        if not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("Force sample max_age_s must be finite and positive")
        self._reader = reader
        self.max_age_s = float(max_age_s)
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._latest: ForceSample | None = None
        self._history = deque(maxlen=32)
        self._error: str | None = None
        self._close_error: str | None = None
        self._thread = threading.Thread(
            target=self._poll_loop, name="fts300-serial", daemon=True
        )
        self._thread.start()

    def _poll_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                started = time.monotonic()
                try:
                    timed_read = getattr(self._reader, "read_samples", None)
                    if callable(timed_read):
                        samples = timed_read()
                    else:
                        wrench = np.asarray(self._reader.read(), dtype=np.float32)
                        samples = [ForceSample(wrench.copy(), started, time.monotonic())]
                    if not samples:
                        raise RuntimeError("Force reader returned no samples")
                    for sample in samples:
                        if sample.wrench.shape != (6,) or not np.isfinite(sample.wrench).all():
                            raise RuntimeError("Force reader returned an invalid six-axis wrench")
                except Exception as exc:
                    with self._condition:
                        self._error = str(exc)
                        self._history.clear()
                        self._condition.notify_all()
                    self._stop_event.wait(0.02)
                    continue
                with self._condition:
                    self._latest = samples[-1]
                    self._history.extend(samples)
                    self._error = None
                    self._condition.notify_all()
        finally:
            try:
                self._reader.close()
            except Exception as exc:
                self._close_error = str(exc)
            with self._condition:
                self._condition.notify_all()

    def wait_until_ready(self, timeout_s: float = 1.0) -> None:
        """Wait for the first sample during startup, outside the recording loop."""
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._latest is None and self._error is None:
                if self._stop_event.is_set():
                    raise RuntimeError("Force sampling is closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Timed out waiting for the first force sample")
                self._condition.wait(remaining)
        self.read_sample()

    def read_sample(self) -> ForceSample:
        with self._condition:
            if self._stop_event.is_set():
                raise RuntimeError("Force sampling is closed")
            if self._error is not None:
                raise RuntimeError(f"Background force read failed: {self._error}")
            sample = self._latest
            if sample is None:
                raise RuntimeError("No force sample is available yet")
            age_s = time.monotonic() - sample.read_started_monotonic
            if age_s > self.max_age_s:
                raise RuntimeError(
                    f"Force sample is stale ({age_s * 1000:.1f} ms > "
                    f"{self.max_age_s * 1000:.1f} ms)"
                )
            return replace(sample, wrench=sample.wrench.copy())

    def read(self) -> np.ndarray:
        return self.read_sample().wrench

    def read_sample_at(self, target_monotonic: float, max_skew_s: float) -> ForceSample:
        """Select history for recording; latest-sample checks still guard control."""
        if not np.isfinite(target_monotonic) or not np.isfinite(max_skew_s) or max_skew_s <= 0:
            raise ValueError("Force recording time and skew must be finite")
        self.read_sample()
        with self._condition:
            if self._error is not None or self._stop_event.is_set():
                raise RuntimeError("Force sample became unavailable")
            candidates = [sample for sample in self._history
                          if abs(sample.timestamp_monotonic-target_monotonic) <= max_skew_s]
            if not candidates:
                raise RuntimeError("No force sample near recording reference")
            sample = min(candidates, key=lambda value: abs(value.timestamp_monotonic-target_monotonic))
            return replace(sample, wrench=sample.wrench.copy())

    def close(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        cancel = getattr(self._reader, "cancel_read", None)
        if cancel is not None and self._thread.is_alive():
            cancel()
        # Account for all configured serial attempts if cancellation interrupts
        # only one pySerial read. The worker closes the port before it exits.
        attempts = max(1, int(getattr(self._reader, "retries", 1)))
        timeout_s = float(getattr(self._reader, "timeout_s", 0.1))
        self._thread.join(timeout=max(1.0, attempts * (timeout_s + 0.02) + 0.5,
                                      float(getattr(self._reader, "close_timeout_s", 0))))
        if self._thread.is_alive():
            raise RuntimeError("Force sampling thread did not exit; serial port is still in use")
        if self._close_error is not None:
            raise RuntimeError(f"Failed to close the force serial port: {self._close_error}")


class RobotiqFTS300ModbusSerial:
    """Read a Robotiq FTS-300-S wrench over Modbus RTU serial.

    This follows the previously tested local code path:
    minimalmodbus.Instrument(serial_port, slaveaddress=9).read_registers(180, 6)
    with force registers scaled by 1/100 N and torque registers by 1/1000 Nm.
    """

    def __init__(
        self,
        port: str | None = None,
        slave_address: int = 9,
        register_address: int = 180,
        function_code: int = 3,
        baudrate: int = 19200,
        bytesize: int = 8,
        parity: str = "N",
        stopbits: int = 1,
        timeout_s: float = 0.1,
        wakeup: bool = True,
        retries: int = 3,
    ) -> None:
        try:
            import minimalmodbus
            import serial
            import serial.tools.list_ports
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading Robotiq FTS-300-S over Modbus RTU requires "
                "minimalmodbus and pyserial. Install them with: "
                "pip install minimalmodbus pyserial"
            ) from exc

        self._minimalmodbus = minimalmodbus
        self._serial = serial
        self.port = port or self._find_serial_port(serial)
        if self.port is None:
            raise RuntimeError(
                "No USB serial port found for Robotiq FTS-300-S. Pass "
                "--force-serial-port /dev/ttyUSBX explicitly."
            )

        self.slave_address = int(slave_address)
        self.register_address = int(register_address)
        self.function_code = int(function_code)
        self.timeout_s = float(timeout_s)
        self.retries = max(1, int(retries))

        if wakeup:
            self._wakeup_serial_adapter(
                port=self.port,
                baudrate=baudrate,
                bytesize=bytesize,
                parity=parity,
                stopbits=stopbits,
                timeout_s=timeout_s,
            )

        instrument = minimalmodbus.Instrument(self.port, self.slave_address)
        instrument.serial.baudrate = baudrate
        instrument.serial.bytesize = bytesize
        instrument.serial.parity = parity
        instrument.serial.stopbits = stopbits
        instrument.serial.timeout = timeout_s
        instrument.mode = minimalmodbus.MODE_RTU
        instrument.clear_buffers_before_each_transaction = True
        instrument.close_port_after_each_call = False
        self._instrument = instrument
        self._lock = threading.Lock()
        print(
            "Robotiq FTS-300-S Modbus RTU "
            f"port={self.port}, slave={self.slave_address}, "
            f"register={self.register_address}, function={self.function_code}, "
            f"baudrate={baudrate}, timeout={timeout_s}s"
        )

    def read(self) -> np.ndarray:
        with self._lock:
            last_exc = None
            for _ in range(self.retries):
                try:
                    registers = self._instrument.read_registers(
                        self.register_address,
                        6,
                        functioncode=self.function_code,
                    )
                    break
                except Exception as exc:
                    last_exc = exc
                    time.sleep(0.02)
            else:
                raise RuntimeError(
                    "No response from Robotiq FTS-300-S over Modbus RTU. "
                    f"port={self.port}, slave={self.slave_address}, "
                    f"register={self.register_address}, function={self.function_code}, "
                    f"timeout={self.timeout_s}s. Check --force-serial-port, USB/RS485 "
                    "wiring, sensor power, slave address, baudrate, and permissions. "
                    f"Original error: {last_exc}"
                ) from last_exc
        values = [
            self._to_int16(registers[0]) / 100.0,
            self._to_int16(registers[1]) / 100.0,
            self._to_int16(registers[2]) / 100.0,
            self._to_int16(registers[3]) / 1000.0,
            self._to_int16(registers[4]) / 1000.0,
            self._to_int16(registers[5]) / 1000.0,
        ]
        return np.asarray(values, dtype=np.float32)

    def close(self) -> None:
        serial_obj = getattr(self._instrument, "serial", None)
        if serial_obj is not None and getattr(serial_obj, "is_open", False):
            serial_obj.close()

    def cancel_read(self) -> None:
        """Interrupt a pending serial read when its sampling worker is stopped."""
        serial_obj = getattr(self._instrument, "serial", None)
        cancel = getattr(serial_obj, "cancel_read", None)
        if cancel is not None and getattr(serial_obj, "is_open", False):
            try:
                cancel()
            except OSError:
                # The worker may have finished and closed the descriptor already.
                pass

    @staticmethod
    def _find_serial_port(serial_module) -> str | None:
        for port_info in serial_module.tools.list_ports.comports():
            port_name = port_info.device
            description = port_info.description or ""
            if "USB" in description or "ttyUSB" in port_name or "ttyACM" in port_name:
                return port_name
        return None

    def _wakeup_serial_adapter(
        self,
        port: str,
        baudrate: int,
        bytesize: int,
        parity: str,
        stopbits: int,
        timeout_s: float,
    ) -> None:
        try:
            with self._serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=bytesize,
                parity=parity,
                stopbits=stopbits,
                timeout=timeout_s,
            ) as serial_port:
                serial_port.write(bytes([0xFF]) * 50)
                serial_port.flush()
                time.sleep(0.02)
        except OSError:
            # The following minimalmodbus read will surface the real connection error.
            pass

    @staticmethod
    def _to_int16(value: int) -> int:
        value = int(value) & 0xFFFF
        return value - 0x10000 if value & 0x8000 else value


class RobotiqFTS300ModbusTCP:
    """Read a Robotiq FTS-300-S wrench from Modbus TCP registers."""

    def __init__(
        self,
        host: str,
        port: int = 502,
        unit_id: int = 9,
        register_address: int = 0,
        function: ModbusFunction = "input",
        value_format: ModbusValueFormat = "int32",
        word_order: ModbusWordOrder = "big",
        scale: float = 1.0,
        timeout_s: float = 1.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.unit_id = int(unit_id)
        self.register_address = int(register_address)
        self.function = function
        self.value_format = value_format
        self.word_order = word_order
        self.scale = float(scale)
        self.timeout_s = float(timeout_s)
        self._transaction_id = 0
        self._lock = threading.Lock()

    def read(self) -> np.ndarray:
        registers = self._read_registers(
            address=self.register_address,
            quantity=self._register_count(),
        )
        return self._decode_wrench(registers)

    def _register_count(self) -> int:
        if self.value_format == "int16":
            return 6
        if self.value_format in ("int32", "float32"):
            return 12
        raise ValueError(f"Unsupported Modbus value format: {self.value_format}")

    def _function_code(self) -> int:
        if self.function == "holding":
            return 3
        if self.function == "input":
            return 4
        raise ValueError(f"Unsupported Modbus function: {self.function}")

    def _next_transaction_id(self) -> int:
        self._transaction_id = (self._transaction_id + 1) % 0x10000
        return self._transaction_id

    def _read_registers(self, address: int, quantity: int) -> list[int]:
        if not 1 <= quantity <= 125:
            raise ValueError(f"Invalid Modbus quantity: {quantity}")

        with self._lock:
            transaction_id = self._next_transaction_id()
            function_code = self._function_code()
            pdu = struct.pack(">BHH", function_code, address, quantity)
            header = struct.pack(">HHHB", transaction_id, 0, len(pdu) + 1, self.unit_id)
            request = header + pdu

            try:
                with socket.create_connection(
                    (self.host, self.port),
                    timeout=self.timeout_s,
                ) as sock:
                    sock.settimeout(self.timeout_s)
                    sock.sendall(request)
                    response_header = self._recv_exact(sock, 7)
                    resp_tid, protocol_id, length, unit_id = struct.unpack(
                        ">HHHB",
                        response_header,
                    )
                    if resp_tid != transaction_id:
                        raise RuntimeError(
                            f"Modbus transaction mismatch: expected {transaction_id}, got {resp_tid}"
                        )
                    if protocol_id != 0:
                        raise RuntimeError(f"Invalid Modbus protocol id: {protocol_id}")
                    if unit_id != self.unit_id:
                        raise RuntimeError(
                            f"Modbus unit mismatch: expected {self.unit_id}, got {unit_id}"
                        )

                    response_pdu = self._recv_exact(sock, length - 1)
            except OSError as exc:
                raise RuntimeError(
                    f"Failed to read Robotiq FTS-300-S Modbus TCP at "
                    f"{self.host}:{self.port}: {exc}"
                ) from exc

        if not response_pdu:
            raise RuntimeError("Empty Modbus response from Robotiq FTS-300-S")

        response_function = response_pdu[0]
        if response_function & 0x80:
            exception_code = response_pdu[1] if len(response_pdu) > 1 else None
            raise RuntimeError(
                f"Modbus exception for function {function_code}: {exception_code}"
            )
        if response_function != function_code:
            raise RuntimeError(
                f"Unexpected Modbus function: expected {function_code}, got {response_function}"
            )

        byte_count = response_pdu[1]
        payload = response_pdu[2:]
        expected_bytes = quantity * 2
        if byte_count != expected_bytes or len(payload) != expected_bytes:
            raise RuntimeError(
                f"Unexpected Modbus payload size: byte_count={byte_count}, "
                f"payload={len(payload)}, expected={expected_bytes}"
            )

        return list(struct.unpack(f">{quantity}H", payload))

    def _recv_exact(self, sock: socket.socket, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise RuntimeError("Modbus connection closed while reading response")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _decode_wrench(self, registers: list[int]) -> np.ndarray:
        if self.value_format == "int16":
            values = [self._to_int16(register) for register in registers[:6]]
        else:
            values = []
            for i in range(0, 12, 2):
                pair = registers[i : i + 2]
                if self.word_order == "little":
                    pair = [pair[1], pair[0]]
                raw_bytes = struct.pack(">HH", pair[0], pair[1])
                if self.value_format == "int32":
                    values.append(struct.unpack(">i", raw_bytes)[0])
                elif self.value_format == "float32":
                    values.append(struct.unpack(">f", raw_bytes)[0])
                else:
                    raise ValueError(
                        f"Unsupported Modbus value format: {self.value_format}"
                    )

        return (np.asarray(values[:6], dtype=np.float32) * self.scale).astype(np.float32)

    @staticmethod
    def _to_int16(value: int) -> int:
        return value - 0x10000 if value & 0x8000 else value


RobotiqFTS300Modbus = RobotiqFTS300ModbusTCP
