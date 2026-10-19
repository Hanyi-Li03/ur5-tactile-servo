"""Module to control Robotiq's grippers - tested with HAND-E.

Taken from https://github.com/githubuser0xFFFF/py_robotiq_gripper/blob/master/src/robotiq_gripper.py
"""

import socket
import threading
import time
from enum import Enum
from typing import OrderedDict, Tuple, Union
import rtde_control

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
    
def main():
    # test open and closing the gripper
    gripper = RobotiqGripper()
    gripper.connect(hostname="192.168.1.10", port=63352)
    # gripper.activate()
    print(gripper.get_current_position())
    gripper.move(20, 255, 1)
    time.sleep(0.2)
    print(gripper.get_current_position())
    gripper.move(65, 255, 1)
    time.sleep(0.2)
    print(gripper.get_current_position())
    gripper.move(20, 255, 1)
    gripper.disconnect()


if __name__ == "__main__":
    main()
