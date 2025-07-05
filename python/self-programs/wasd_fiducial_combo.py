# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.

"""WASD driving of robot + live AprilTag scanning (shows tag IDs in message log)."""
import curses
import logging
import math
import os
import signal
import sys
import threading
import time

from PIL import Image
import numpy as np

import bosdyn.client.util
from bosdyn.api import image_pb2, geometry_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient

try:
    from pupil_apriltags import Detector
except ImportError:
    print("Please install pupil-apriltags (`pip install pupil-apriltags`)")
    sys.exit(1)

LOGGER = logging.getLogger()
VELOCITY_BASE_SPEED = 0.5  # m/s
VELOCITY_BASE_ANGULAR = 0.8  # rad/sec
VELOCITY_CMD_DURATION = 0.6  # seconds
COMMAND_INPUT_RATE = 0.1

class ExitCheck(object):
    def __init__(self):
        self._kill_now = False
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        signal.signal(signal.SIGINT, self._sigterm_handler)

    def __enter__(self): return self
    def __exit__(self, *_): return False
    def _sigterm_handler(self, *_): self._kill_now = True
    def request_exit(self): self._kill_now = True
    @property
    def kill_now(self): return self._kill_now

class WasdInterface(object):
    def __init__(self, robot):
        self._robot = robot
        self._lease_client = robot.ensure_client(LeaseClient.default_service_name)
        self._power_client = robot.ensure_client(PowerClient.default_service_name)
        self._robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self._robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._lock = threading.Lock()
        self._messages = ['', '', '']
        self._exit_check = None
        self._lease_keepalive = None

    def start(self):
        self._lease_keepalive = LeaseKeepAlive(self._lease_client, must_acquire=True, return_at_exit=True)
        self._robot_id = self._robot.get_id()

    def shutdown(self):
        if self._lease_keepalive:
            self._lease_keepalive.shutdown()

    def add_message(self, msg_text):
        with self._lock:
            self._messages = [msg_text] + self._messages[:-1]

    def message(self, idx):
        with self._lock:
            return self._messages[idx]

    def drive(self, stdscr):
        with ExitCheck() as self._exit_check:
            stdscr.nodelay(True)
            stdscr.resize(26, 100)
            stdscr.refresh()
            try:
                while not self._exit_check.kill_now:
                    self._drive_draw(stdscr)
                    key = stdscr.getch()
                    self._drive_cmd(key)
                    time.sleep(COMMAND_INPUT_RATE)
            except Exception as e:
                self.shutdown()
                raise

    def _drive_draw(self, stdscr):
        stdscr.clear()
        stdscr.resize(26, 100)
        stdscr.addstr(0, 0, f'Spot WASD + AprilTag Scanner')
        for i in range(3):
            stdscr.addstr(2 + i, 2, self.message(i))
        stdscr.addstr(6, 0, 'Commands: [wasd]=move, [q/e]=turn, [f]=stand, [v]=sit, [p]=power, [ESC]=stop, [TAB]=quit')
        stdscr.refresh()

    def _drive_cmd(self, key):
        try:
            if key == 27: self._stop()
            elif key == ord('\t'): self._quit_program()
            elif key == ord('p'): self._toggle_power()
            elif key == ord('f'): self._stand()
            elif key == ord('v'): self._sit()
            elif key == ord('w'): self._velocity_cmd_helper('forward', v_x=VELOCITY_BASE_SPEED)
            elif key == ord('s'): self._velocity_cmd_helper('back', v_x=-VELOCITY_BASE_SPEED)
            elif key == ord('a'): self._velocity_cmd_helper('left', v_y=VELOCITY_BASE_SPEED)
            elif key == ord('d'): self._velocity_cmd_helper('right', v_y=-VELOCITY_BASE_SPEED)
            elif key == ord('q'): self._velocity_cmd_helper('turn_left', v_rot=VELOCITY_BASE_ANGULAR)
            elif key == ord('e'): self._velocity_cmd_helper('turn_right', v_rot=-VELOCITY_BASE_ANGULAR)
        except Exception:
            pass

    def _quit_program(self):
        self._sit()
        if self._exit_check is not None:
            self._exit_check.request_exit()

    def _stand(self):
        self._robot_command_client.robot_command(RobotCommandBuilder.synchro_stand_command())
    def _sit(self):
        self._robot_command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
    def _stop(self):
        self._robot_command_client.robot_command(RobotCommandBuilder.stop_command())
    def _toggle_power(self):
        power_client = self._power_client
        state = self._robot_state_client.get_robot_state().power_state.motor_power_state
        if state == 1:  # STATE_ON
            power_client.power_off()
            self.add_message("Powering off")
        else:
            power_client.power_on()
            self.add_message("Powering on")
    def _velocity_cmd_helper(self, desc='', v_x=0.0, v_y=0.0, v_rot=0.0):
        self._robot_command_client.robot_command(
            RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot),
            end_time_secs=time.time() + VELOCITY_CMD_DURATION)

class FiducialScanner(threading.Thread):
    def __init__(self, robot, wasd_interface, options):
        super().__init__(daemon=True)
        self.robot = robot
        self.options = options
        self.image_client = robot.ensure_client(ImageClient.default_service_name)
        self.running = True
        self.last_tag_ids = []
        self._lock = threading.Lock()
        self._image_sources = [
            src.name for src in self.image_client.list_image_sources()
            if src.image_type == image_pb2.ImageSource.IMAGE_TYPE_VISUAL and 'depth' not in src.name
        ]
        self.wasd_interface = wasd_interface

    def stop(self):
        self.running = False

    def run(self):
        detector = Detector(families='tag36h11')
        while self.running:
            tag_ids_found = set()
            for src_name in self._image_sources:
                try:
                    img_req = build_image_request(src_name, quality_percent=60, image_format=image_pb2.Image.FORMAT_RAW)
                    image_response = self.image_client.get_image([img_req])[0]
                    width = image_response.shot.image.cols
                    height = image_response.shot.image.rows
                    image_grey = np.array(
                        Image.frombytes('P', (int(width), int(height)), data=image_response.shot.image.data, decoder_name='raw'))
                    detections = detector.detect(image_grey)
                    for det in detections:
                        tag_ids_found.add(det.tag_id)
                except Exception:
                    continue
            tag_ids_list = sorted(list(tag_ids_found))
            if tag_ids_list != self.last_tag_ids:
                if tag_ids_list:
                    msg = f"AprilTags detected: {tag_ids_list}"
                else:
                    msg = "No AprilTags detected."
                self.wasd_interface.add_message(msg)
                self.last_tag_ids = tag_ids_list
            time.sleep(0.7)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--hostname', required=False, help='Spot robot IP/hostname')
    options = parser.parse_args()

    sdk = create_standard_sdk('WASDFiducialCombo')
    robot = sdk.create_robot(options.hostname)
    try:
        bosdyn.client.util.authenticate(robot)
        robot.start_time_sync()
    except RpcError as err:
        LOGGER.error('Failed to communicate with robot: %s', err)
        return False

    wasd_interface = WasdInterface(robot)
    wasd_interface.start()

    tag_scanner = FiducialScanner(robot, wasd_interface, options)
    tag_scanner.start()

    try:
        os.environ.setdefault('ESCDELAY', '0')
        curses.wrapper(wasd_interface.drive)
    finally:
        wasd_interface.shutdown()
        tag_scanner.stop()

    return True

if __name__ == '__main__':
    if not main():
        sys.exit(1)
