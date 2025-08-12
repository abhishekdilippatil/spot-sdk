# Copyright (c) 2023 Boston Dynamics, Inc.
# All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Detect and follow AprilTags with manual WASD control + prompt-to-follow.
   - Minimal log spam: single-line status that overwrites itself.
   - Persistent AprilTag detector to avoid resource issues.
   - No image rotation; PnP uses consistent intrinsics.
   - Obstacle avoidance ON by default (override with --avoid-obstacles false).
   - Keys: W/A/S/D, Q/E, R(sweep), F/C(prompt), M(cancel follow), ESC(exit).
"""

import logging
import math
import signal
import sys
import threading
import time

import cv2
import numpy as np
from pupil_apriltags import Detector as apriltag

import bosdyn.client
import bosdyn.client.util
import bosdyn.client.robot_command
from bosdyn import geometry
from bosdyn.api import geometry_pb2, image_pb2, trajectory_pb2
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import RpcError, create_standard_sdk
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, VISION_FRAME_NAME, get_a_tform_b,
                                         get_vision_tform_body)
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseClient
from bosdyn.client.math_helpers import Quat
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_id import RobotIdClient
from bosdyn.client.robot_state import RobotStateClient

# cross-platform key reading
import select
try:
    import msvcrt
    _WIN = True
except Exception:
    import tty, termios
    _WIN = False

# ---------------------------- Config & logging ----------------------------

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

VELOCITY_BASE_SPEED   = 0.5   # m/s (overridden by CLI)
VELOCITY_BASE_ANGULAR = 0.8   # rad/s (overridden by CLI)
VELOCITY_CMD_DURATION = 0.6   # seconds (overridden by CLI)
BODY_LENGTH = 1.1             # meters

# ---------------------------- Single-line status printer ----------------------------

class StatusPrinter:
    """Thread-safe single-line status updates that overwrite in terminal."""
    def __init__(self, width=120):
        self._lock = threading.Lock()
        self._width = width

    def update(self, msg: str):
        with self._lock:
            print((msg or "").ljust(self._width), end='\r', flush=True)

    def clear(self):
        with self._lock:
            print(' ' * self._width, end='\r', flush=True)

    def note(self, msg: str):
        """Print a one-off message on a new line, clearing the status row first."""
        with self._lock:
            print(' ' * self._width, end='\r', flush=True)
            print(msg, flush=True)

status = StatusPrinter()

# ---------------------------- Core fiducial follower ----------------------------

class FollowFiducial(object):
    """Detect and follow AprilTags with manual interaction."""
    TAG_TIMEOUT = 4.0  # seconds before a tag is considered lost

    def __init__(self, robot, options):
        # Mode: 'manual' -> WASD, 'prompt' -> found tag prompt, 'following' -> trajectory
        self.mode = 'manual'

        self._robot = robot
        self._robot_id = robot.ensure_client(RobotIdClient.default_service_name).get_id(timeout=0.4)
        self._power_client = robot.ensure_client(PowerClient.default_service_name)
        self._image_client = robot.ensure_client(ImageClient.default_service_name)
        self._robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)

        # Options
        self._tag_size_mm = float(options.tag_size_mm)
        self._tag_offset = float(options.distance_margin) + BODY_LENGTH / 2.0
        self._limit_speed = options.limit_speed
        self._avoid_obstacles = options.avoid_obstacles
        self._vx = float(options.vel_speed)
        self._vrot = float(options.vel_ang)
        self._vel_duration = float(options.vel_duration)

        # Speed limits if limit_speed=True
        self._max_x_vel = 0.2
        self._max_y_vel = 0.2
        self._max_ang_vel = 0.2

        # Tolerances
        self._x_eps = .05
        self._y_eps = .05
        self._angle_eps = .005

        # State
        self._standup = True
        self._movement_on = True
        self._powered_on = False
        self._running = True

        self._intrinsics = None
        self._dist_coeffs = np.zeros((5, 1), dtype=float)   # updated if available
        self._camera_tform_body = None
        self._body_tform_world = None
        self._current_tag_world_pose = np.array([])
        self._angle_desired = None
        self._image = dict()
        self._previous_source = None

        # Image sources (visual only)
        self._source_names = [
            src.name for src in self._image_client.list_image_sources()
            if (src.image_type == image_pb2.ImageSource.IMAGE_TYPE_VISUAL and 'depth' not in src.name)
        ]
        status.note(f"Camera sources: {self._source_names}")

        # Persistent detector (avoid reconstructing every frame)
        self._detector = apriltag(families='tag36h11')

        # Tag tracking across frames
        self._all_tags = {}  # {tag_id: {'info': info, 'last_seen': timestamp, 'sweep_seen': bool}}
        self._sweeping = False
        self._tag_lock = threading.Lock()

        # Prompt results shared across threads
        self._pending_lock = threading.Lock()
        self._pending_detections = None

    @property
    def robot_state(self):
        return self._robot_state_client.get_robot_state()

    @property
    def image(self):
        return self._image

    @property
    def image_sources_list(self):
        return self._source_names

    # -------------------- Lifecycle --------------------

    def start_detection_thread(self):
        self._detection_active = True
        self._detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._detection_thread.start()

    def stop_detection_thread(self):
        self._detection_active = False
        if hasattr(self, '_detection_thread'):
            self._detection_thread.join(timeout=1.0)

    def start(self):
        self._robot.time_sync.wait_for_sync()
        if self._standup:
            self.power_on()
            blocking_stand(self._robot_command_client)
            time.sleep(.35)

        self.start_detection_thread()
        try:
            while self._running:
                time.sleep(0.05)
        finally:
            self.stop_detection_thread()
            if self._powered_on:
                self.power_off()
            status.clear()

    def stop(self):
        self._running = False

    def power_on(self):
        self._robot.power_on()
        self._powered_on = True
        status.note(f'Powered On {self._robot.is_powered_on()}')

    def power_off(self):
        self._robot.power_off()
        status.note(f'Powered Off {not self._robot.is_powered_on()}')

    # -------------------- Motion helpers --------------------

    def _velocity_command(self, desc='', v_x=0.0, v_y=0.0, v_rot=0.0, duration=None):
        if duration is None:
            duration = self._vel_duration
        if not (self._movement_on and self._powered_on):
            status.update(f"Cannot {desc} — movement disabled or motors off.")
            return
        mobility_params = self.set_mobility_params()
        vel_cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=v_x, v_y=v_y, v_rot=v_rot,
            frame_name=BODY_FRAME_NAME,
            params=mobility_params,
            body_height=0.0,
            locomotion_hint=spot_command_pb2.HINT_AUTO,
        )
        try:
            self._robot_command_client.robot_command(command=vel_cmd, end_time_secs=time.time() + duration)
            time.sleep(duration)
        finally:
            try:
                self._robot_command_client.robot_command(command=RobotCommandBuilder.stop_command())
            except bosdyn.client.robot_command.ExpiredError:
                pass

    def rotate_in_place(self, angle_rad, angular_speed=0.5):
        """Rotate body yaw by angle_rad (positive=CCW) at angular_speed."""
        dur = max(abs(angle_rad) / max(angular_speed, 1e-3), 0.01)
        w = angular_speed if angle_rad >= 0.0 else -angular_speed
        self._velocity_command('rotate_in_place', v_rot=w, duration=dur)

    # WASD bindings (use CLI magnitudes)
    def move_forward(self):  self._velocity_command('move_forward',  v_x= self._vx)
    def move_backward(self): self._velocity_command('move_backward', v_x=-self._vx)
    def strafe_left(self):   self._velocity_command('strafe_left',   v_y= self._vx)
    def strafe_right(self):  self._velocity_command('strafe_right',  v_y=-self._vx)
    def turn_left(self):     self._velocity_command('turn_left',     v_rot= self._vrot)
    def turn_right(self):    self._velocity_command('turn_right',    v_rot=-self._vrot)

    # -------------------- Detection thread --------------------

    def _detection_loop(self):
        while getattr(self, '_detection_active', False):
            try:
                bboxes, tag_ids, source_name = self.image_to_bounding_box()
            except Exception as e:
                status.update(f"Detection loop: {type(e).__name__} {e}")
                time.sleep(0.2)
                continue

            if source_name:
                if tag_ids:
                    status.update(f"{source_name}: {len(tag_ids)} tag(s) detected")
                else:
                    status.update(f"No tags in {source_name}")

            now = time.time()
            infos = self._pnp_each(bboxes, tag_ids)
            with self._tag_lock:
                for info in infos:
                    tid = info['id']
                    self._all_tags.setdefault(tid, {'info': info, 'last_seen': now, 'sweep_seen': False})
                    self._all_tags[tid]['info'] = info
                    self._all_tags[tid]['last_seen'] = now
                    if self._sweeping:
                        self._all_tags[tid]['sweep_seen'] = True
                if not self._sweeping:
                    to_remove = [tid for tid, v in self._all_tags.items()
                                 if now - v['last_seen'] > self.TAG_TIMEOUT]
                    for tid in to_remove:
                        del self._all_tags[tid]

            # Prompt user only in manual mode
            if bboxes and tag_ids and self.mode == 'manual':
                self._previous_source = source_name
                self.on_tags_detected(bboxes, tag_ids, source_name)

            time.sleep(0.05)

    # -------------------- AprilTag + PnP + transforms --------------------

    def _announce_tag_prompt(self, infos):
        ids = [i['id'] for i in infos]
        closest = min(infos, key=lambda x: x['dist'])
        status.note(f"\n[AprilTag] Detected IDs: {ids}\n"
                    f"Closest: ID={closest['id']} at ~{closest['dist']:.2f} m.\n"
                    f"Press [F] to FOLLOW closest, or [C] to CONTINUE manual driving.")

    def _pnp_each(self, bboxes, tag_ids):
        out = []
        if self._intrinsics is None:
            return out
        K = self.make_camera_matrix(self._intrinsics)
        dist = self._dist_coeffs  # zeros or parsed
        S = self._tag_size_mm  # mm edge length of the black square
        # tl, tr, bl, br
        obj_points = np.array([[0, 0, 0],
                               [S, 0, 0],
                               [0, S, 0],
                               [S, S, 0]], dtype=np.float32)
        for idx, tag_id in enumerate(tag_ids):
            img_points = self.corners_to_img_points(bboxes[idx])  # tl,tr,bl,br
            try:
                ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist,
                                              flags=cv2.SOLVEPNP_IPPE_SQUARE)
            except Exception as e:
                status.update(f"PnP(id={tag_id}): {type(e).__name__} {e}")
                continue
            if not ok:
                continue
            tz = float(tvec[2][0])
            if tz <= 0:  # behind camera / degenerate
                continue
            tx, ty = float(tvec[0][0]), float(tvec[1][0])
            dist_m = math.sqrt(tx*tx + ty*ty + tz*tz) / 1000.0  # mm -> m
            out.append({'id': int(tag_id), 'tvec': tvec, 'dist': dist_m})
        return out

    def on_tags_detected(self, bboxes, tag_ids, source_name):
        if self.mode != 'manual':
            return
        infos = self._pnp_each(bboxes, tag_ids)
        if not infos:
            return
        with self._pending_lock:
            self._pending_detections = infos
        self.mode = 'prompt'
        self._announce_tag_prompt(infos)

    def follow_detected_closest(self):
        with self._pending_lock:
            pend = self._pending_detections
        if not pend:
            status.note("No pending detections.")
            return
        info = min(pend, key=lambda x: x['dist'])
        vision_pos = self.compute_fiducial_in_world_frame(info['tvec'])
        fid_vec = geometry_pb2.Vec3(x=vision_pos[0], y=vision_pos[1], z=vision_pos[2])
        status.note(f"\nFollowing AprilTag ID {info['id']} ...")
        self.mode = 'following'
        self.go_to_tag(fid_vec)
        status.note(f"Reached tag ID {info['id']}. Manual mode restored.")
        with self._pending_lock:
            self._pending_detections = None
        self.mode = 'manual'

    def dismiss_prompt(self):
        status.note("Continuing manual driving.")
        with self._pending_lock:
            self._pending_detections = None
        self.mode = 'manual'

    def cancel_follow(self):
        try:
            self._robot_command_client.robot_command(command=RobotCommandBuilder.stop_command())
        finally:
            self.mode = 'manual'
            status.note("Follow cancelled. Manual mode.")

    # -------------------- Image capture + detection --------------------

    def image_to_bounding_box(self):
        # Prefer last source; otherwise round-robin through all sources
        for i in range(len(self._source_names) + 1):
            if i == 0:
                if self._previous_source is not None:
                    source_name = self._previous_source
                else:
                    continue
            elif self._source_names[i - 1] == self._previous_source:
                continue
            else:
                source_name = self._source_names[i - 1]
            try:
                img_req = build_image_request(source_name, quality_percent=100,
                                              image_format=image_pb2.Image.FORMAT_RAW)
                image_response = self._image_client.get_image([img_req])
                shot = image_response[0].shot
                src = image_response[0].source

                # Transforms
                self._camera_tform_body = get_a_tform_b(shot.transforms_snapshot,
                                                        shot.frame_name_image_sensor,
                                                        BODY_FRAME_NAME)
                self._body_tform_world = get_a_tform_b(shot.transforms_snapshot,
                                                       BODY_FRAME_NAME, VISION_FRAME_NAME)
                # Intrinsics & (optional) distortion
                self._intrinsics = src.pinhole.intrinsics
                self._dist_coeffs = self._get_distortion_coeffs_safe(src)

                w, h = shot.image.cols, shot.image.rows
                expected = int(w) * int(h)
                if len(shot.image.data) != expected:
                    # Unexpected buffer size — skip frame (common if stream hiccups)
                    raise ValueError(f"buffer={len(shot.image.data)} != w*h={expected}")

                bboxes, tag_ids = self.detect_fiducial_in_image(shot.image, (w, h), source_name)
                if bboxes:
                    return bboxes, tag_ids, source_name
            except Exception as e:
                status.update(f"Detect({source_name}): {type(e).__name__} {e}")
        return [], [], None

    def detect_fiducial_in_image(self, image, dim, source_name):
        try:
            w, h = int(dim[0]), int(dim[1])
            # Raw grayscale buffer -> (H, W) uint8. NO rotation (keep K valid).
            img = np.frombuffer(image.data, dtype=np.uint8).reshape(h, w)
            detections = self._detector.detect(img)

            bboxes, tag_ids = [], []
            for det in detections:
                # det.corners: tl, tr, br, bl
                bbox = det.corners.astype(np.float32)
                tag_id = int(det.tag_id)
                bboxes.append(bbox)
                tag_ids.append(tag_id)
                # Optional visual outline for preview (white)
                cv2.polylines(img, [bbox.astype(np.int32)], True, 255, 2)

            self._image[source_name] = img  # stored unrotated (consistent with intrinsics)
            return bboxes, tag_ids
        except Exception as e:
            status.update(f"Detect({source_name}): {type(e).__name__} {e}")
            return [], []

    @staticmethod
    def corners_to_img_points(bbox):
        """Map pupil_apriltags corners (tl,tr,br,bl) -> [tl,tr,bl,br]."""
        tl, tr, br, bl = bbox[0], bbox[1], bbox[2], bbox[3]
        return np.array([tl, tr, bl, br], dtype=np.float32)

    # -------------------- World transform & goal --------------------

    def compute_fiducial_in_world_frame(self, tvec):
        # Convert mm -> m, then camera->body->vision
        fid_cam = np.array([float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])]) / 1000.0
        body_pt = (self._camera_tform_body.inverse()).transform_point(*fid_cam)
        vis_pt = self._body_tform_world.inverse().transform_point(*body_pt)
        return vis_pt

    def go_to_tag(self, fiducial_rt_world):
        self._current_tag_world_pose, self._angle_desired = self.offset_tag_pose(
            fiducial_rt_world, self._tag_offset)
        mobility_params = self.set_mobility_params()
        tag_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=self._current_tag_world_pose[0], goal_y=self._current_tag_world_pose[1],
            goal_heading=self._angle_desired, frame_name=VISION_FRAME_NAME, params=mobility_params,
            body_height=0.0, locomotion_hint=spot_command_pb2.HINT_AUTO)
        end_time = 30.0
        if self._movement_on and self._powered_on:
            self._robot_command_client.robot_command(command=tag_cmd, end_time_secs=time.time() + end_time)
            start = time.time()
            while (self.mode == 'following') and (not self.final_state()) and ((time.time() - start) < end_time):
                time.sleep(.25)

    def final_state(self):
        robot_state = get_vision_tform_body(self.robot_state.kinematic_state.transforms_snapshot)
        robot_angle = robot_state.rot.to_yaw()
        if self._current_tag_world_pose.size != 0:
            x_dist = abs(self._current_tag_world_pose[0] - robot_state.x)
            y_dist = abs(self._current_tag_world_pose[1] - robot_state.y)
            angle = abs(self._angle_desired - robot_angle)
            if ((x_dist < self._x_eps) and (y_dist < self._y_eps) and (angle < self._angle_eps)):
                return True
        return False

    def get_desired_angle(self, xhat):
        zhat = [0.0, 0.0, 1.0]
        yhat = np.cross(zhat, xhat)
        mat = np.array([xhat, yhat, zhat]).transpose()
        return Quat.from_matrix(mat).to_yaw()

    def offset_tag_pose(self, object_rt_world, dist_margin=1.0):
        robot_rt_world = get_vision_tform_body(self.robot_state.kinematic_state.transforms_snapshot)
        v = np.array([object_rt_world.x - robot_rt_world.x,
                      object_rt_world.y - robot_rt_world.y,
                      0.0])
        n = np.linalg.norm(v)
        if n < 1e-6: n = 1e-6
        u = v / n
        heading = self.get_desired_angle(u)
        goto = np.array([object_rt_world.x - u[0]*dist_margin,
                         object_rt_world.y - u[1]*dist_margin])
        return goto, heading

    def set_mobility_params(self):
        obstacles = spot_command_pb2.ObstacleParams(
            disable_vision_body_obstacle_avoidance=True,
            disable_vision_foot_obstacle_avoidance=True,
            disable_vision_foot_constraint_avoidance=True,
            obstacle_avoidance_padding=.001)
        body_control = self.set_default_body_control()
        if self._limit_speed:
            speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
                linear=Vec2(x=self._max_x_vel, y=self._max_y_vel), angular=self._max_ang_vel))
            if not self._avoid_obstacles:
                return spot_command_pb2.MobilityParams(
                    obstacle_params=obstacles, vel_limit=speed_limit, body_control=body_control,
                    locomotion_hint=spot_command_pb2.HINT_AUTO)
            else:
                return spot_command_pb2.MobilityParams(
                    vel_limit=speed_limit, body_control=body_control,
                    locomotion_hint=spot_command_pb2.HINT_AUTO)
        elif not self._avoid_obstacles:
            return spot_command_pb2.MobilityParams(
                obstacle_params=obstacles, body_control=body_control,
                locomotion_hint=spot_command_pb2.HINT_AUTO)
        else:
            return None

    @staticmethod
    def set_default_body_control():
        footprint_R_body = geometry.EulerZXY()
        position = geometry_pb2.Vec3(x=0.0, y=0.0, z=0.0)
        rotation = footprint_R_body.to_quaternion()
        pose = geometry_pb2.SE3Pose(position=position, rotation=rotation)
        point = trajectory_pb2.SE3TrajectoryPoint(pose=pose)
        traj = trajectory_pb2.SE3Trajectory(points=[point])
        return spot_command_pb2.BodyControlParams(base_offset_rt_footprint=traj)

    @staticmethod
    def make_camera_matrix(ints):
        # Correct pinhole matrix (single skew parameter)
        return np.array([[ints.focal_length.x, ints.skew.x,         ints.principal_point.x],
                         [0.0,                 ints.focal_length.y, ints.principal_point.y],
                         [0.0,                 0.0,                 1.0]], dtype=float)

    @staticmethod
    def _get_distortion_coeffs_safe(src):
        """Attempt to read distortion from calibration; fallback to zeros."""
        try:
            intr = src.pinhole.intrinsics
            dnames = ['k1','k2','p1','p2','k3']
            vals = []
            for name in dnames:
                vals.append(float(getattr(intr, name)) if hasattr(intr, name) else 0.0)
            return np.array(vals, dtype=float).reshape(-1,1)
        except Exception:
            return np.zeros((5,1), dtype=float)

    # -------------------- Prompt helpers --------------------

    def prompt_tag_choice(self, tag_ids):
        if not tag_ids:
            status.note("No tags detected.")
            return None
        if len(tag_ids) == 1:
            choice = input(f"One tag detected: {tag_ids[0]}. Follow? [y/n]: ").strip().lower()
            if choice == 'y':
                return tag_ids[0]
            return None
        print(f"Detected tags: {tag_ids}")
        while True:
            choice = input("Enter the ID of the tag to follow, or 'n' to cancel: ").strip()
            if choice == 'n':
                return None
            try:
                chosen_tag_id = int(choice)
                if chosen_tag_id in tag_ids:
                    return chosen_tag_id
                print("Invalid tag ID. Try again.")
            except Exception:
                print("Invalid input. Enter a numeric tag ID or 'n'.")

# ---------------------------- Keyboard controller ----------------------------

class KeyboardController(threading.Thread):
    def __init__(self, follower: FollowFiducial):
        super().__init__(daemon=True)
        self.f = follower
        self._stop = False
        if not _WIN:
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

    def stop(self):
        self._stop = True

    def _read_key(self):
        if _WIN:
            if msvcrt.kbhit():
                return msvcrt.getwch()
            return None
        else:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:
                return sys.stdin.read(1)
            return None

    def run(self):
        try:
            while not self._stop:
                ch = self._read_key()
                if not ch:
                    time.sleep(0.03)
                    continue
                if ch == '\x1b':  # ESC
                    self.f.stop()
                    self.stop()
                    continue
                if self.f.mode == 'prompt':
                    if ch.lower() == 'f':
                        self.f.follow_detected_closest()
                    elif ch.lower() == 'c':
                        self.f.dismiss_prompt()
                    continue
                if self.f.mode == 'following':
                    if ch.lower() == 'm':
                        self.f.cancel_follow()
                    continue
                # Manual mode controls
                if   ch.lower() == 'w': self.f.move_forward()
                elif ch.lower() == 's': self.f.move_backward()
                elif ch.lower() == 'a': self.f.strafe_left()
                elif ch.lower() == 'd': self.f.strafe_right()
                elif ch.lower() == 'q': self.f.turn_left()
                elif ch.lower() == 'e': self.f.turn_right()
                elif ch.lower() == 'r':
                    status.note("Search mode: sweeping yaw...")
                    self.f._previous_source = None
                    self.f._sweeping = True
                    self.f.rotate_in_place(angle_rad=-math.radians(45), angular_speed=0.5)
                    time.sleep(0.3)
                    self.f.rotate_in_place(angle_rad= math.radians(90), angular_speed=0.5)
                    time.sleep(0.3)
                    self.f.rotate_in_place(angle_rad=-math.radians(45), angular_speed=0.5)
                    self.f._sweeping = False
        finally:
            if not _WIN:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

# ---------------------------- Preview windows (optional) ----------------------------

class DisplayImagesAsync(object):
    def __init__(self, fiducial_follower):
        self._fiducial_follower = fiducial_follower
        self._thread = None
        self._started = False
        self._sources = []

    def get_image(self):
        images = self._fiducial_follower.image
        image_by_source = []
        for s_name in self._sources:
            if s_name in images:
                image_by_source.append(images[s_name])
            else:
                image_by_source.append(np.array([]))
        return image_by_source

    def start(self):
        if self._started:
            return None
        self._sources = self._fiducial_follower.image_sources_list
        self._started = True
        self._thread = threading.Thread(target=self.update, daemon=True)
        self._thread.start()
        return self

    def update(self):
        while self._started:
            images = self.get_image()
            for i, image in enumerate(images):
                if image.size != 0:
                    h, w = image.shape[:2]
                    resized = cv2.resize(image, (int(w * .5), int(h * .5)), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow(self._sources[i], resized)
                    cv2.moveWindow(self._sources[i], int(i * w * .5), 0)
                    cv2.waitKey(1)

    def stop(self):
        self._started = False
        cv2.destroyAllWindows()

# ---------------------------- Exit helper ----------------------------

class Exit(object):
    def __init__(self):
        self._kill_now = False
        signal.signal(signal.SIGTERM, self._sigterm_handler)

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        return False

    def _sigterm_handler(self, _signum, _frame):
        self._kill_now = True

    @property
    def kill_now(self):
        return self._kill_now

# ---------------------------- Main ----------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--distance-margin', default=.25, type=float,
                        help='Distance [m] to stop from the tag (added to half body length).')
    parser.add_argument('--tag-size-mm', default=146.0, type=float,
                        help='Black-square edge size of the AprilTag in millimeters.')
    parser.add_argument('--limit-speed', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Limit maximum speed (default: True).')
    parser.add_argument('--avoid-obstacles', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Enable obstacle avoidance (default: True).')
    parser.add_argument('--show-preview', action='store_true', default=False,
                        help='Show camera preview windows (default: False)')
    parser.add_argument('--vel-speed', type=float, default=VELOCITY_BASE_SPEED,
                        help='Linear speed for WASD moves [m/s].')
    parser.add_argument('--vel-ang', type=float, default=VELOCITY_BASE_ANGULAR,
                        help='Angular speed for Q/E turns [rad/s].')
    parser.add_argument('--vel-duration', type=float, default=VELOCITY_CMD_DURATION,
                        help='Duration for each WASD velocity command [s].')

    options = parser.parse_args()

    sdk = create_standard_sdk('FollowFiducialClient')
    robot = sdk.create_robot(options.hostname)

    fiducial_follower = None
    image_viewer = None
    kb = None

    try:
        with Exit():
            bosdyn.client.util.authenticate(robot)
            robot.start_time_sync()
            assert not robot.is_estopped(), 'Robot is estopped. Use E-Stop client to configure.'

            # Print explicit obstacle avoidance state once.
            oa_state = "ON" if options.avoid_obstacles else "OFF"
            status.note(f"Obstacle avoidance: {oa_state}")

            fiducial_follower = FollowFiducial(robot, options)
            kb = KeyboardController(fiducial_follower)
            kb.start()
            time.sleep(.1)

            if sys.platform.lower() != 'darwin' and options.show_preview:
                image_viewer = DisplayImagesAsync(fiducial_follower)
                image_viewer.start()

            lease_client = robot.ensure_client(LeaseClient.default_service_name)
            with bosdyn.client.lease.LeaseKeepAlive(
                lease_client, must_acquire=True, return_at_exit=True
            ):
                fiducial_follower.start()

    except RpcError as err:
        # One-off critical error: keep as a proper log
        LOGGER.error('Failed to communicate with robot: %s', err)
    finally:
        if image_viewer is not None:
            image_viewer.stop()
        if kb is not None:
            kb.stop()
            try:
                kb.join(timeout=1.0)
            except Exception:
                pass
        status.clear()
    return False

if __name__ == '__main__':
    if not main():
        sys.exit(1)