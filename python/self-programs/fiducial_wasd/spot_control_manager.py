# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

import curses
import io
import logging
import math
import os
import signal
import sys
import threading
import time
from collections import OrderedDict
import locale
locale.setlocale(locale.LC_ALL, '')  # better width handling for unicode/wide chars
from types import SimpleNamespace

import bosdyn.api.basic_command_pb2 as basic_command_pb2
import bosdyn.api.spot.robot_command_pb2 as spot_command_pb2
import bosdyn.api.robot_state_pb2 as robot_state_proto
import bosdyn.api.power_pb2 as PowerServiceProto
import bosdyn
import bosdyn.client.util
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.exceptions import ResponseError, RpcError
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.sdk import create_standard_sdk
from bosdyn.client.time_sync import TimeSyncError
from bosdyn.client.async_tasks import AsyncGRPCTask, AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.util import duration_str, format_metric, secs_to_hms

from fiducial_follow import FollowFiducial, DisplayImagesAsync

VELOCITY_BASE_SPEED = 0.5  # m/s
VELOCITY_BASE_ANGULAR = 0.8  # rad/sec
VELOCITY_CMD_DURATION = 0.6  # seconds
COMMAND_INPUT_RATE = 0.1

#pylint: disable=no-member
LOGGER = logging.getLogger()

def _safe_addstr(win, y, x, s):
    """Write string safely within window bounds, clipping if needed."""
    if s is None:
        return
    try:
        max_y, max_x = win.getmaxyx()
    except curses.error:
        return
    if y < 0 or x < 0 or y >= max_y:
        return
    # Leave the last column free to avoid bottom-right ERR
    avail = max_x - x - 1
    if avail <= 0:
        return
    try:
        win.addnstr(y, x, str(s), avail)
    except curses.error:
        # Ignore sporadic errors (e.g., during console resize)
        pass

def _ensure_min_size(stdscr, min_rows=17, min_cols=600):
    """Show a single-line warning if terminal is too small."""
    try:
        max_y, max_x = stdscr.getmaxyx()
    except curses.error:
        return False
    if max_y < min_rows or max_x < min_cols:
        stdscr.clear()
        _safe_addstr(stdscr, 0, 0,
            f"Terminal too small ({max_y}x{max_x}). "
            f"Resize to at least {min_rows}x{min_cols}.")
        stdscr.refresh()
        return False
    return True

def _try_resize(stdscr, rows=40, cols=300):
    """Best-effort resize; safe to call on platforms that support it."""
    try:
        curses.resizeterm(rows, cols)  # safer than stdscr.resize on Windows
        stdscr.clear()
    except curses.error:
        # Ignore if not supported / console can't be resized
        pass

class ExitCheck(object):
    """A class to help exiting a loop, also capturing SIGTERM to exit the loop."""

    def __init__(self):
        self._kill_now = False
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        signal.signal(signal.SIGINT, self._sigterm_handler)

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        return False

    def _sigterm_handler(self, _signum, _frame):
        self._kill_now = True

    def request_exit(self):
        """Manually trigger an exit (rather than sigterm/sigint)."""
        self._kill_now = True

    @property
    def kill_now(self):
        """Return the status of the exit checker indicating if it should exit."""
        return self._kill_now


class CursesHandler(logging.Handler):
    """logging handler which puts messages into the curses interface"""

    def __init__(self, spot_interface):
        super(CursesHandler, self).__init__()
        self._spot_interface = spot_interface

    def emit(self, record):
        msg = record.getMessage()
        msg = msg.replace('\n', ' ').replace('\r', '')
        self._spot_interface.add_message(f'{record.levelname:s} {msg:s}')


class AsyncRobotState(AsyncPeriodicQuery):
    """Grab robot state."""

    def __init__(self, robot_state_client):
        super(AsyncRobotState, self).__init__('robot_state', robot_state_client, LOGGER,
                                              period_sec=0.2)

    def _start_query(self):
        return self._client.get_robot_state_async()

class KyeboardSpotManager(object):
    """A curses interface for estop, lease and motor power of the robot."""

    def __init__(self, robot):
        self._robot = robot
        # Create clients -- do not use the for communication yet.
        self._lease_client = robot.ensure_client(LeaseClient.default_service_name)
        try:
            self._estop_client = self._robot.ensure_client(EstopClient.default_service_name)
            self._estop_endpoint = EstopEndpoint(self._estop_client, 'GNClient', 9.0)
        except:
            # Not the estop.
            self._estop_client = None
            self._estop_endpoint = None
        self._power_client = robot.ensure_client(PowerClient.default_service_name)
        self._robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self._robot_state_task = AsyncRobotState(self._robot_state_client)
        self._async_tasks = AsyncTasks([self._robot_state_task])
        self._lock = threading.Lock()
        self._command_dictionary = {
            27: self._stop,  # ESC key
            ord('\t'): self._quit_program,
            ord('T'): self._toggle_time_sync,
            ord(' '): self._toggle_estop,
            ord('r'): self._self_right,
            ord('P'): self._toggle_power,
            ord('p'): self._toggle_power,
            ord('v'): self._sit,
            ord('b'): self._battery_change_pose,
            ord('f'): self._stand,
            ord('w'): self._move_forward,
            ord('s'): self._move_backward,
            ord('a'): self._strafe_left,
            ord('d'): self._strafe_right,
            ord('q'): self._turn_left,
            ord('e'): self._turn_right,
            ord('l'): self._toggle_lease,
            ord('i'): self._toggle_camera_viewer,  # start/stop camera streaming
            ord('o'): self._toggle_fiducial_follow
        }
        self._locked_messages = ['', '', '']  # string: displayed message for user
        self._estop_keepalive = None
        self._exit_check = None
        self._fiducial = None
        self._fid_thread = None
        self._viewer = None

        # Stuff that is set in start()
        self._robot_id = None
        self._lease_keepalive = None

    def start(self):
        """Begin communication with the robot."""
        # Construct our lease keep-alive object, which begins RetainLease calls in a thread.
        self._lease_keepalive = LeaseKeepAlive(self._lease_client, must_acquire=True,
                                               return_at_exit=True)

        self._robot_id = self._robot.get_id()
        if self._estop_endpoint is not None:
            self._estop_endpoint.force_simple_setup(
            )  # Set this endpoint as the robot's sole estop.

    def shutdown(self):
        """Release control of robot as gracefully as possible."""
        LOGGER.info('Shutting down KeyboardSpotManager.')
        if self._estop_keepalive:
            # This stops the check-in thread but does not stop the robot.
            self._estop_keepalive.shutdown()
        if self._lease_keepalive:
            self._lease_keepalive.shutdown()
        
        # stop camera viewer if running
        if getattr(self, "_viewer", None) is not None:
            try:
                self._viewer.stop()
            except Exception:
                pass
            self._viewer = None

        # stop fiducial follower if running
        if getattr(self, "_fiducial", None) is not None:
            try:
                self._fiducial.stop()
            except Exception:
                pass
            if getattr(self, "_fid_thread", None) is not None:
                try:
                    self._fid_thread.join(timeout=2.0)
                except Exception:
                    pass
                self._fid_thread = None

    def flush_and_estop_buffer(self, stdscr):
        """Manually flush the curses input buffer but trigger any estop requests (space)"""
        key = ''
        while key != -1:
            key = stdscr.getch()
            if key == ord(' '):
                self._toggle_estop()

    def add_message(self, msg_text):
        """Display the given message string to the user in the curses interface."""
        with self._lock:
            self._locked_messages = [msg_text] + self._locked_messages[:-1]

    def message(self, idx):
        """Grab one of the 3 last messages added."""
        with self._lock:
            return self._locked_messages[idx]

    @property
    def robot_state(self):
        """Get latest robot state proto."""
        return self._robot_state_task.proto

    def drive(self, stdscr):
        """User interface to control the robot via the passed-in curses screen interface object."""
        with ExitCheck() as self._exit_check:
            curses_handler = CursesHandler(self)
            curses_handler.setLevel(logging.INFO)
            LOGGER.addHandler(curses_handler)

            curses.noecho()
            curses.cbreak()
            stdscr.nodelay(True)  # Don't block for user input.
            stdscr.keypad(True)
            # Sanity check: ensure we got a curses window, not a shadowed name
            assert hasattr(stdscr, 'addstr'), f"stdscr is not a window (got {type(stdscr)})"
            stdscr.refresh()

            try:
                while not self._exit_check.kill_now:
                    if not _ensure_min_size(stdscr, 17, 60):
                        time.sleep(0.2)
                        continue
                    self._async_tasks.update()
                    self._drive_draw(stdscr, self._lease_keepalive)

                    try:
                        cmd = stdscr.getch()
                        # Do not queue up commands on client
                        self.flush_and_estop_buffer(stdscr)
                        self._drive_cmd(cmd)
                        time.sleep(COMMAND_INPUT_RATE)
                    except Exception:
                        # On robot command fault, sit down safely before killing the program.
                        self._safe_power_off()
                        time.sleep(2.0)
                        raise

            finally:
                LOGGER.removeHandler(curses_handler)

    def _drive_draw(self, stdscr, lease_keep_alive):
        """Draw the interface screen at each update."""
        stdscr.clear()  # clear screen
        _safe_addstr(stdscr, 0, 0, f'{self._robot_id.nickname:20s} {self._robot_id.serial_number}')
        _safe_addstr(stdscr, 1, 0, self._lease_str(lease_keep_alive))
        _safe_addstr(stdscr, 2, 0, self._battery_str())
        _safe_addstr(stdscr, 3, 0, self._estop_str())
        _safe_addstr(stdscr, 4, 0, self._power_state_str())
        _safe_addstr(stdscr, 5, 0, self._time_sync_str())
        for i in range(3):
            _safe_addstr(stdscr, 7 + i, 2, self.message(i))
        _safe_addstr(stdscr, 10, 0, 'Commands: [TAB]: quit                                              ')
        _safe_addstr(stdscr, 11, 0, '          [T]: Time-sync, [SPACE]: Estop, [P]: Power               ')
        _safe_addstr(stdscr, 12, 0, '          [f]: Stand, [r]: Self-right, [wasd]: Directional strafing')
        _safe_addstr(stdscr, 13, 0, '          [v]: Sit, [b]: Battery-change, [qe]: Turning             ')
        _safe_addstr(stdscr, 14, 0, '          [i]: Toggle camera view (OpenCV windows), [ESC]: Stop    ')
        _safe_addstr(stdscr, 15, 0, '          [o]: Toggle fiducial follow [l]: Return/Acquire lease    ')
        _safe_addstr(stdscr, 16, 0, '')

        stdscr.refresh()

    def _drive_cmd(self, key):
        """Run user commands at each update."""
        try:
            cmd_function = self._command_dictionary[key]
            cmd_function()

        except KeyError:
            if key and key != -1 and key < 256:
                self.add_message(f'Unrecognized keyboard command: \'{chr(key)}\'')

    def _try_grpc(self, desc, thunk):
        try:
            return thunk()
        except (ResponseError, RpcError, LeaseBaseError) as err:
            self.add_message(f'Failed {desc}: {err}')
            return None

    def _try_grpc_async(self, desc, thunk):

        def on_future_done(fut):
            try:
                fut.result()
            except (ResponseError, RpcError, LeaseBaseError) as err:
                self.add_message(f'Failed {desc}: {err}')
                return None

        future = thunk()
        future.add_done_callback(on_future_done)

    def _quit_program(self):
        self._sit()
        if self._exit_check is not None:
            self._exit_check.request_exit()

    def _toggle_time_sync(self):
        if self._robot.time_sync.stopped:
            self._robot.start_time_sync()
        else:
            self._robot.time_sync.stop()

    def _toggle_estop(self):
        """toggle estop on/off. Initial state is ON"""
        if self._estop_client is not None and self._estop_endpoint is not None:
            if not self._estop_keepalive:
                self._estop_keepalive = EstopKeepAlive(self._estop_endpoint)
            else:
                self._try_grpc('stopping estop', self._estop_keepalive.stop)
                self._estop_keepalive.shutdown()
                self._estop_keepalive = None

    def _toggle_lease(self):
        """toggle lease acquisition. Initial state is acquired"""
        if self._lease_client is not None:
            if self._lease_keepalive is None:
                self._lease_keepalive = LeaseKeepAlive(self._lease_client, must_acquire=True,
                                                       return_at_exit=True)
            else:
                self._lease_keepalive.shutdown()
                self._lease_keepalive = None

    def _toggle_camera_viewer(self):
        """Start/stop OpenCV camera streaming independently of fiducial follow."""
        try:
            self._ensure_fiducial()  # provides image_client + rotate_image
            if self._viewer is None:
                self._viewer = DisplayImagesAsync(self._fiducial)  # starts with defaults
                self._viewer.start()
                self.add_message("Camera viewer: STARTED")
            else:
                self._viewer.stop()
                self._viewer = None
                self.add_message("Camera viewer: STOPPED")
        except Exception as e:
            self.add_message(f"Viewer error: {e}")

    def _toggle_fiducial_follow(self):
        """Start/stop fiducial follower in its own thread."""
        try:
            self._ensure_fiducial()
            # Start if not running
            if self._fid_thread is None or not self._fid_thread.is_alive():
                self._fid_thread = threading.Thread(target=self._fiducial.start, daemon=True)
                self._fid_thread.start()
                self.add_message("Fiducial follow: STARTED")
            else:
                # Stop if running
                self._fiducial.stop()
                try:
                    self._fid_thread.join(timeout=2.0)
                except Exception:
                    pass
                self._fid_thread = None
                self.add_message("Fiducial follow: STOPPED")
        except Exception as e:
            self.add_message(f"Fiducial error: {e}")

    def _start_robot_command(self, desc, command_proto, end_time_secs=None):

        def _start_command():
            self._robot_command_client.robot_command(command=command_proto,
                                                     end_time_secs=end_time_secs)

        self._try_grpc(desc, _start_command)

    def _self_right(self):
        self._start_robot_command('self_right', RobotCommandBuilder.selfright_command())

    def _battery_change_pose(self):
        # Default HINT_RIGHT, maybe add option to choose direction?
        self._start_robot_command(
            'battery_change_pose',
            RobotCommandBuilder.battery_change_pose_command(
                dir_hint=basic_command_pb2.BatteryChangePoseCommand.Request.HINT_RIGHT))

    def _sit(self):
        self._start_robot_command('sit', RobotCommandBuilder.synchro_sit_command())

    def _stand(self):
        self._start_robot_command('stand', RobotCommandBuilder.synchro_stand_command())

    def _stop(self):
        self._start_robot_command('stop', RobotCommandBuilder.stop_command())

    def _move_forward(self):
        self._velocity_cmd_helper('move_forward', v_x=VELOCITY_BASE_SPEED)

    def _move_backward(self):
        self._velocity_cmd_helper('move_backward', v_x=-VELOCITY_BASE_SPEED)

    def _strafe_left(self):
        self._velocity_cmd_helper('strafe_left', v_y=VELOCITY_BASE_SPEED)

    def _strafe_right(self):
        self._velocity_cmd_helper('strafe_right', v_y=-VELOCITY_BASE_SPEED)

    def _turn_left(self):
        self._velocity_cmd_helper('turn_left', v_rot=VELOCITY_BASE_ANGULAR)

    def _turn_right(self):
        self._velocity_cmd_helper('turn_right', v_rot=-VELOCITY_BASE_ANGULAR)

    def _velocity_cmd_helper(self, desc='', v_x=0.0, v_y=0.0, v_rot=0.0):
        self._start_robot_command(
            desc, RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot),
            end_time_secs=time.time() + VELOCITY_CMD_DURATION)

    def _return_to_origin(self):
        self._start_robot_command(
            'fwd_and_rotate',
            RobotCommandBuilder.synchro_se2_trajectory_point_command(
                goal_x=0.0, goal_y=0.0, goal_heading=0.0, frame_name=ODOM_FRAME_NAME, params=None,
                body_height=0.0, locomotion_hint=spot_command_pb2.HINT_SPEED_SELECT_TROT),
            end_time_secs=time.time() + 20)

    def _toggle_power(self):
        power_state = self._power_state()
        if power_state is None:
            self.add_message('Could not toggle power because power state is unknown')
            return

        if power_state == robot_state_proto.PowerState.STATE_OFF:
            self._try_grpc_async('powering-on', self._request_power_on)
        else:
            self._try_grpc('powering-off', self._safe_power_off)

    def _request_power_on(self):
        request = PowerServiceProto.PowerCommandRequest.REQUEST_ON
        return self._power_client.power_command_async(request)

    def _safe_power_off(self):
        self._start_robot_command('safe_power_off', RobotCommandBuilder.safe_power_off_command())

    def _power_state(self):
        state = self.robot_state
        if not state:
            return None
        return state.power_state.motor_power_state

    def _lease_str(self, lease_keep_alive):
        if lease_keep_alive is None:
            alive = 'STOPPED'
            lease = 'RETURNED'
        else:
            try:
                _lease = lease_keep_alive.lease_wallet.get_lease()
                lease = f'{_lease.lease_proto.resource}:{_lease.lease_proto.sequence}'
            except bosdyn.client.lease.Error:
                lease = '...'
            if lease_keep_alive.is_alive():
                alive = 'RUNNING'
            else:
                alive = 'STOPPED'
        return f'Lease {lease} THREAD:{alive}'

    def _power_state_str(self):
        power_state = self._power_state()
        if power_state is None:
            return ''
        state_str = robot_state_proto.PowerState.MotorPowerState.Name(power_state)
        return f'Power: {state_str[6:]}'  # get rid of STATE_ prefix

    def _estop_str(self):
        if not self._estop_client:
            thread_status = 'NOT ESTOP'
        else:
            thread_status = 'RUNNING' if self._estop_keepalive else 'STOPPED'
        estop_status = '??'
        state = self.robot_state
        if state:
            for estop_state in state.estop_states:
                if estop_state.type == estop_state.TYPE_SOFTWARE:
                    estop_status = estop_state.State.Name(estop_state.state)[6:]  # s/STATE_//
                    break
        return f'Estop {estop_status} (thread: {thread_status})'

    def _time_sync_str(self):
        if not self._robot.time_sync:
            return 'Time sync: (none)'
        if self._robot.time_sync.stopped:
            status = 'STOPPED'
            exception = self._robot.time_sync.thread_exception
            if exception:
                status = f'{status} Exception: {exception}'
        else:
            status = 'RUNNING'
        try:
            skew = self._robot.time_sync.get_robot_clock_skew()
            if skew:
                skew_str = f'offset={duration_str(skew)}'
            else:
                skew_str = '(Skew undetermined)'
        except (TimeSyncError, RpcError) as err:
            skew_str = f'({err})'
        return f'Time sync: {status} {skew_str}'

    def _battery_str(self):
        if not self.robot_state:
            return ''
        battery_state = self.robot_state.battery_states[0]
        status = battery_state.Status.Name(battery_state.status)
        status = status[7:]  # get rid of STATUS_ prefix
        pct = ''
        try:
            if battery_state.HasField('charge_percentage'):
                pct = f'{battery_state.charge_percentage.value:.0f}'
        except Exception:
             # Fallback if HasField isn't available
            v = getattr(getattr(battery_state, 'charge_percentage', None), 'value', None)
            if v is not None:
                pct = f'{v:.0f}'
        time_left = ''
        if battery_state.estimated_runtime:
            time_left = f'({secs_to_hms(battery_state.estimated_runtime.seconds)})'
        return f'Battery: {status} {pct + "%" if pct!= ""else""} {time_left}'.strip()
    
    def _fiducial_options(self):
        # Defaults: use Spot's world-object service so you don't need apriltag installed.
        return SimpleNamespace(
            distance_margin=0.15,
            limit_speed=True,
            avoid_obstacles=True,
            use_world_objects=True,
        )

    def _ensure_fiducial(self):
        if self._fiducial is None:
            self._fiducial = FollowFiducial(self._robot, self._fiducial_options())


def _setup_logging(verbose):
    """Log to file at debug level, and log to console at INFO or DEBUG (if verbose).

    Returns the stream/console logger so that it can be removed when in curses mode.
    """
    LOGGER.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Save log messages to file spot_control_manager.log for later debugging.
    file_handler = logging.FileHandler('spot_control_manager.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    LOGGER.addHandler(file_handler)

    # The stream handler is useful before and after the application is in curses-mode.
    if verbose:
        stream_level = logging.DEBUG
    else:
        stream_level = logging.INFO

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(log_formatter)
    LOGGER.addHandler(stream_handler)
    return stream_handler

def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--time-sync-interval-sec',
                        help='The interval (seconds) that time-sync estimate should be updated.',
                        type=float)
    options = parser.parse_args()

    stream_handler = _setup_logging(options.verbose)

    # Create robot object.
    sdk = create_standard_sdk('SpotManagerClient')
    robot = sdk.create_robot(options.hostname)
    try:
        bosdyn.client.util.authenticate(robot)
        robot.start_time_sync(options.time_sync_interval_sec)
    except RpcError as err:
        LOGGER.error('Failed to communicate with robot: %s', err)
        return False

    spot_interface = KyeboardSpotManager(robot)
    try:
        spot_interface.start()
    except (ResponseError, RpcError) as err:
        LOGGER.error('Failed to initialize robot communication: %s', err)
        return False

    # LOGGER.removeHandler(stream_handler)  # Don't use stream handler in curses mode.

    try:
        try:
            # Prevent curses from introducing a 1-second delay for ESC key
            os.environ.setdefault('ESCDELAY', '0')
            # Run spot interface in curses mode, then restore terminal config.
            curses.wrapper(spot_interface.drive)
        finally:
            # Restore stream handler to show any exceptions or final messages.
            # LOGGER.addHandler(stream_handler)
            pass
    except Exception as e:
        LOGGER.exception('Spot Manager has thrown an error: [%r] %s', e, e)
    finally:
        # Do any final cleanup steps.
        spot_interface.shutdown()

    return True


if __name__ == '__main__':
    if not main():
        sys.exit(1)