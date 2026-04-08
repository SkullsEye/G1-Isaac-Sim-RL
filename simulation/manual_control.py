#!/usr/bin/env python3
import time
import threading
from pynput import keyboard
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

class KeyboardController:
    def __init__(self):
        # [x_vel, y_vel, yaw_vel, height]
        self.control_params = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'h': 0.0}
        self.gripper_open = True
        self.running = True
        self.lock = threading.Lock()
        
        print("=== Manual Control ===")
        print("WASD: Move Base")
        print("Space: Toggle Gripper")
        print("Q: Quit")

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            k = key.char.lower()
            with self.lock:
                if k == 'w': self.control_params['x'] += 0.1
                elif k == 's': self.control_params['x'] -= 0.1
                elif k == 'a': self.control_params['y'] += 0.1
                elif k == 'd': self.control_params['y'] -= 0.1
                elif k == 'q': self.running = False
        except AttributeError:
            if key == keyboard.Key.space:
                with self.lock:
                    self.gripper_open = not self.gripper_open
                    print(f"Gripper: {'OPEN' if self.gripper_open else 'CLOSED'}")

    def get_cmd(self):
        with self.lock:
            # Format: [x, y, yaw, height, gripper_state]
            # Gripper: 0.0 = Open, 1.0 = Closed (approx)
            g_val = 0.0 if self.gripper_open else 1.0
            cmd = [self.control_params['x'], -self.control_params['y'], 
                   -self.control_params['yaw'], 0.8 + self.control_params['h'], g_val]
            return str(cmd)

if __name__ == "__main__":
    ChannelFactoryInitialize(1)
    pub = ChannelPublisher("rt/run_command/cmd", String_)
    pub.Init()
    
    # Try creating a publisher for the gripper specifically if needed
    # pub_g = ChannelPublisher("rt/dex1/cmd", String_)
    # pub_g.Init()

    ctrl = KeyboardController()
    last_cmd = ""

    while ctrl.running:
        time.sleep(0.05)
        cmd_str = ctrl.get_cmd()
        if cmd_str != last_cmd:
            pub.Write(String_(data=cmd_str))
            # pub_g.Write(String_(data=cmd_str)) 
            last_cmd = cmd_str
            print(f"Sent: {cmd_str}")
