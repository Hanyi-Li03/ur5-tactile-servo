"""Control the OnRobot RG2 through UR RTDE custom scripts.

Original driver reference:
https://github.com/githubuser0xFFFF/py_robotiq_gripper/blob/master/src/robotiq_gripper.py
"""

import time
import math
import threading


class AsyncRG2Gripper:
    """Run RG2 scripts outside the robot RPC thread, with one pending target."""

    def __init__(self, gripper, *, timeout_s: float = 10.0):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("RG2 operation timeout must be finite and positive")
        self._gripper = gripper
        self._timeout_s = timeout_s
        self._condition = threading.Condition()
        self._pending = None
        self._active = None
        self._completed = None
        self._started = 0.0
        self._error = None
        self._closed = False
        self._thread = None

    def move(self, target_width: float, force: float) -> bool:
        if not math.isfinite(target_width) or not math.isfinite(force):
            raise ValueError("RG2 width and force must be finite")
        target = (max(0.0, min(110.0, float(target_width))),
                  max(4.0, min(40.0, float(force))))
        with self._condition:
            if self._closed:
                raise RuntimeError("RG2 command worker is closed")
            self._check_timeout()
            if self._error is not None:
                raise RuntimeError(self._error)
            if target == self._active or (self._active is None and target == self._completed):
                self._pending = None
            else:
                self._pending = target
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="rg2-command", daemon=True
                )
                self._thread.start()
            self._condition.notify_all()
        return True

    def _check_timeout(self):
        if self._active is not None and time.monotonic() - self._started > self._timeout_s:
            self._error = "RG2 script timed out; restart the hardware node before moving"
            self._pending = None

    def status(self) -> dict:
        with self._condition:
            self._check_timeout()
            return {
                "busy": self._active is not None or self._pending is not None,
                "error": self._error,
            }

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                target = self._pending
                self._pending = None
                self._active = target
                self._started = time.monotonic()
            try:
                if not self._gripper.move(*target):
                    raise RuntimeError("RG2 script execution failed")
            except Exception as exc:
                with self._condition:
                    self._error = str(exc)
                    self._pending = None
            else:
                with self._condition:
                    self._check_timeout()
                    self._completed = target
            finally:
                with self._condition:
                    self._active = None
                    self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=self._timeout_s)
            if self._thread.is_alive():
                raise RuntimeError("RG2 native call has not stopped; hardware node must exit")


class RG2Gripper:
    def __init__(self, robot, cmd_lock, host_name="192.168.1.102", port_num=29999):
        # init variables
        self.host_name = host_name
        self.port_num = port_num
        self.cmd_lock = cmd_lock
        self.robot = robot

        print("gripper init")

    def _build_rg2_function_body(self, target_width: float, target_force: float) -> str:
        """
        Only function BODY. The outer function wrapper is added by
        sendCustomScriptFunction(function_name, body).
        """
        target_width = max(0.0, min(110.0, float(target_width)))
        target_force = max(4.0, min(40.0, float(target_force)))

        return f"""
target_width={target_width}
target_force={target_force}
payload=1.0
set_payload1=False
depth_compensation=False
slave=False

timeout = 0
while get_digital_in(9) == False:
    if timeout > 400:
        break
    end
    timeout = timeout + 1
    sync()
end

def rg2_bit(input):
    msb=65536
    i=0
    output=0
    while i<17:
        set_digital_out(8,True)
        if input>=msb:
            input=input-msb
            set_digital_out(9,False)
        else:
            set_digital_out(9,True)
        end
        sync()
        set_digital_out(8,False)
        sync()
        input=input*2
        output=output*2
        i=i+1
    end
    return output
end

if target_force > 40:
    target_force = 40
end
if target_force < 4:
    target_force = 4
end
if target_width > 110:
    target_width = 110
end
if target_width < 0:
    target_width = 0
end

rg_data = floor(target_width) * 4
rg_data = rg_data + floor(target_force / 2) * 4 * 111

if slave:
    rg_data = rg_data + 16384
end

rg2_bit(rg_data)

timeout = 0
while get_digital_in(9) == True:
    if timeout > 20:
        break
    end
    timeout = timeout + 1
    sync()
end

timeout = 0
while get_digital_in(9) == False:
    if timeout > 400:
        break
    end
    timeout = timeout + 1
    sync()
end
"""

    def move(self, target_width, force):
        tcp_command = self._build_rg2_function_body(target_width, force)
        with self.cmd_lock:
            ok = self.robot.sendCustomScriptFunction("rg2_remote", tcp_command)
        time.sleep(0.008)

        return ok
