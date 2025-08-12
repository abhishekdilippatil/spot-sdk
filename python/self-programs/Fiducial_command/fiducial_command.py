# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

""" Detect and follow AprilTag. """
import logging
import math
import signal
import sys
import threading
import time
import os
from sys import platform

import cv2
import numpy as np
from PIL import Image
from pupil_apriltags import Detector as apriltag

import bosdyn.client
import bosdyn.client.util
import bosdyn.client.robot_command
from bosdyn import geometry
from bosdyn.api import geometry_pb2, image_pb2, trajectory_pb2
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, VISION_FRAME_NAME, get_a_tform_b,
                                         get_vision_tform_body)
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseClient
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_id import RobotIdClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.geometry import EulerZXY
import select
try:
    import msvcrt
    _WIN = True
except Exception:
    import tty, termios
    _WIN = False


LOGGER = logging.getLogger()
VELOCITY_BASE_SPEED   = 0.5   # m/s
VELOCITY_BASE_ANGULAR = 0.8   # rad/s
VELOCITY_CMD_DURATION = 0.6   # seconds
BODY_LENGTH = 1.1

ROT_MAP = {
    'back_fisheye_image':  cv2.ROTATE_90_CLOCKWISE,
    'frontleft_fisheye_image': cv2.ROTATE_180,
    'frontright_fisheye_image': cv2.ROTATE_180,
    'left_fisheye_image':  cv2.ROTATE_90_COUNTERCLOCKWISE,
    'right_fisheye_image': cv2.ROTATE_90_CLOCKWISE,
}

class FollowFiducial(object):
    """ Detect and follow AprilTags with universal tag tracking and robust user interaction. """

    TAG_TIMEOUT = 2.0  # seconds before a tag is considered lost

    def __init__(self, robot, options):
        self.mode = 'manual'
        self._pending_detections = None
        self._last_detected_tag_ids = []
        self._last_chosen_tag_id = None
        self._robot = robot
        self._robot_id = robot.ensure_client(RobotIdClient.default_service_name).get_id(timeout=0.4)
        self._power_client = robot.ensure_client(PowerClient.default_service_name)
        self._image_client = robot.ensure_client(ImageClient.default_service_name)
        self._robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self._tag_offset = float(options.distance_margin) + BODY_LENGTH / 2.0
        self._max_x_vel = 0.2
        self._max_y_vel = 0.2
        self._max_ang_vel = 0.2
        self._standup = True
        self._movement_on = True
        self._limit_speed = options.limit_speed
        self._avoid_obstacles = options.avoid_obstacles
        self._x_eps = .05
        self._y_eps = .05
        self._angle_eps = .005
        self._powered_on = False
        self._attempts = 0
        self._max_attempts = 5
        self._intrinsics = None
        self._camera_tform_body = None
        self._body_tform_world = None
        self._current_tag_world_pose = np.array([])
        self._angle_desired = None
        self._image = dict()
        self._source_names = [
            src.name for src in self._image_client.list_image_sources()
            if (src.image_type == image_pb2.ImageSource.IMAGE_TYPE_VISUAL and 'depth' not in src.name)
        ]
        LOGGER.info(f"Camera sources: {self._source_names}")
        self._camera_to_extrinsics_guess = self.populate_source_dict()
        self._previous_source = None

        # Universal tag tracking
        self._all_tags = {}  # {tag_id: {'info': info, 'last_seen': timestamp, 'sweep_seen': bool}}
        self._sweeping = False
        self._tag_lock = threading.Lock()  # Thread safety for tag dictionary

    @property
    def robot_state(self):
        return self._robot_state_client.get_robot_state()

    @property
    def image(self):
        return self._image

    @property
    def image_sources_list(self):
        return self._source_names

    def populate_source_dict(self):
        camera_to_extrinsics_guess = dict()
        for src in self._source_names:
            camera_to_extrinsics_guess[src] = (False, (None, None))
        return camera_to_extrinsics_guess

    def start_detection_thread(self):
        self._detection_active = True
        self._detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._detection_thread.start()

    def stop_detection_thread(self):
        self._detection_active = False
        if hasattr(self, '_detection_thread'):
            self._detection_thread.join(timeout=1.0)

    def _detection_loop(self):
        while self._detection_active:
            try:
                bboxes, tag_ids, source_name = self.image_to_bounding_box()
            except Exception as e:
                LOGGER.error(f"Image acquisition or detection failed: {e}")
                time.sleep(0.2)
                continue

            now = time.time()
            infos = self._pnp_each(bboxes, tag_ids)
            current_tags = set()
            with self._tag_lock:
                for info in infos:
                    tag_id = info['id']
                    if tag_id not in self._all_tags:
                        self._all_tags[tag_id] = {'info': info, 'last_seen': now, 'sweep_seen': False}
                    else:
                        self._all_tags[tag_id]['info'] = info
                        self._all_tags[tag_id]['last_seen'] = now
                    current_tags.add(tag_id)
                    if self._sweeping:
                        self._all_tags[tag_id]['sweep_seen'] = True
                # Remove tags not seen recently (unless sweeping)
                if not self._sweeping:
                    to_remove = [tid for tid, v in self._all_tags.items()
                                 if now - v['last_seen'] > self.TAG_TIMEOUT]
                    for tid in to_remove:
                        del self._all_tags[tid]
            # Prompt logic (only in manual mode)
            if bboxes and tag_ids and self.mode == 'manual':
                self._previous_source = source_name
                self.on_tags_detected(bboxes, tag_ids, source_name)
            time.sleep(0.1)

    def start(self):
        self._robot.time_sync.wait_for_sync()
        if self._standup:
            self.power_on()
            blocking_stand(self._robot_command_client)
            time.sleep(.35)
        self.start_detection_thread()
        while self._attempts <= self._max_attempts:
            time.sleep(0.02)
        self.stop_detection_thread()
        if self._powered_on:
            self.power_off()

    def power_on(self):
        self._robot.power_on()
        self._powered_on = True
        LOGGER.info(f'Powered On {self._robot.is_powered_on()}')

    def power_off(self):
        self._robot.power_off()
        LOGGER.info(f'Powered Off {not self._robot.is_powered_on()}')

    def _velocity_command(self, desc='', v_x=0.0, v_y=0.0, v_rot=0.0, duration=VELOCITY_CMD_DURATION):
        if not (self._movement_on and self._powered_on):
            LOGGER.info(f"Cannot {desc} — movement disabled or motors off.")
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

    def move_forward(self):  self._velocity_command('move_forward',  v_x= VELOCITY_BASE_SPEED)
    def move_backward(self): self._velocity_command('move_backward', v_x=-VELOCITY_BASE_SPEED)
    def strafe_left(self):   self._velocity_command('strafe_left',   v_y= VELOCITY_BASE_SPEED)
    def strafe_right(self):  self._velocity_command('strafe_right',  v_y=-VELOCITY_BASE_SPEED)
    def turn_left(self):     self._velocity_command('turn_left',     v_rot= VELOCITY_BASE_ANGULAR)
    def turn_right(self):    self._velocity_command('turn_right',    v_rot=-VELOCITY_BASE_ANGULAR)

    def _announce_tag_prompt(self, infos):
        ids = [i['id'] for i in infos]
        closest = min(infos, key=lambda x: x['dist'])
        print("\n[AprilTag] Detected IDs:", ids)
        print(f"Closest: ID={closest['id']} at ~{closest['dist']:.2f} m.")
        print("Press [F] to FOLLOW closest, or [C] to CONTINUE manual driving.")

    def _pnp_each(self, bboxes, tag_ids):
        out = []
        camera = self.make_camera_matrix(self._intrinsics)
        for idx, tag_id in enumerate(tag_ids):
            obj_points, img_points = self.bbox_to_image_object_pts(bboxes[idx])
            try:
                ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera, np.zeros((5, 1)))
            except Exception as e:
                LOGGER.error(f"PnP failed for tag {tag_id}: {e}")
                continue
            if not ok:
                continue
            dist_m = math.sqrt(float(tvec[0][0])**2 + float(tvec[1][0])**2 + float(tvec[2][0])**2) / 1000.0
            out.append({'id': int(tag_id), 'tvec': tvec, 'dist': dist_m})
        return out

    def on_tags_detected(self, bboxes, tag_ids, source_name):
        if self.mode != 'manual':
            return
        infos = self._pnp_each(bboxes, tag_ids)
        if not infos:
            return
        self._pending_detections = infos
        self.mode = 'prompt'
        self._announce_tag_prompt(infos)

    def follow_detected_closest(self):
        if not self._pending_detections:
            print("No pending detections.")
            return
        info = min(self._pending_detections, key=lambda x: x['dist'])
        vision_pos = self.compute_fiducial_in_world_frame(info['tvec'])
        fid_vec = geometry_pb2.Vec3(x=vision_pos[0], y=vision_pos[1], z=vision_pos[2])
        print(f"\nFollowing AprilTag ID {info['id']} ...")
        self.mode = 'following'
        self.go_to_tag(fid_vec)
        print(f"Reached tag ID {info['id']}. Manual mode restored.")
        self._pending_detections = None
        self.mode = 'manual'

    def dismiss_prompt(self):
        print("Continuing manual driving.")
        self._pending_detections = None
        self.mode = 'manual'

    def cancel_follow(self):
        try:
            self._robot_command_client.robot_command(command=RobotCommandBuilder.stop_command())
        finally:
            self.mode = 'manual'
            print("Follow cancelled. Manual mode.")

    def prompt_tag_choice(self, tag_ids):
        """Prompt user to choose a tag from the detected list."""
        if not tag_ids:
            print("No tags detected.")
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

    def sweep_yaw(self, yaw_range=math.radians(90), angular_speed=0.5):
        """Smoothly sweep Spot's body yaw left and right, collecting all AprilTags detected."""
        self._previous_source = None
        self._sweeping = True
        with self._tag_lock:
            for v in self._all_tags.values():
                v['sweep_seen'] = False
        print("Sweeping left...")
        self.rotate_in_place(angle_rad=-yaw_range/2, angular_speed=angular_speed)
        time.sleep(0.5)
        print("Sweeping right...")
        self.rotate_in_place(angle_rad=yaw_range, angular_speed=angular_speed)
        time.sleep(0.5)
        print("Returning to center...")
        self.rotate_in_place(angle_rad=-yaw_range/2, angular_speed=angular_speed)
        time.sleep(0.5)
        self._sweeping = False

        with self._tag_lock:
            sweep_tags = [tid for tid, v in self._all_tags.items() if v.get('sweep_seen')]
        if sweep_tags:
            print("\nSweep complete. Detected AprilTags:")
            for tag_id in sweep_tags:
                info = self._all_tags[tag_id]['info']
                print(f"  Tag ID: {tag_id}, Distance: {info['dist']:.2f} m")
            chosen_tag_id = self.prompt_tag_choice(sweep_tags)
            if chosen_tag_id is not None:
                print(f"User chose tag {chosen_tag_id}. Following...")
                self.follow_tag_by_id(chosen_tag_id)
            else:
                print("No tag chosen. Returning to manual mode.")
                self.mode = 'manual'
        else:
            print("No AprilTags detected during sweep.")
            self.mode = 'manual'

    def follow_tag_by_id(self, tag_id):
        with self._tag_lock:
            tag_entry = self._all_tags.get(tag_id)
        if tag_entry:
            info = tag_entry['info']
            vision_pos = self.compute_fiducial_in_world_frame(info['tvec'])
            fid_vec = geometry_pb2.Vec3(x=vision_pos[0], y=vision_pos[1], z=vision_pos[2])
            self.mode = 'following'
            self.go_to_tag(fid_vec)
            print(f"Reached tag ID {tag_id}. Manual mode restored.")
            self.mode = 'manual'
        else:
            print(f"Tag ID {tag_id} not found in sweep results.")

    def capture_and_save_photos_from_all_cameras(self):
        print("\n Capture photos from all cameras")
        input("Press Enter to take photos from all cameras...")
        images = {}
        for source_name in self._source_names:
            try:
                print(f" Capturing image from {source_name}...")
                img_req = build_image_request(source_name, quality_percent=100,
                                              image_format=image_pb2.Image.FORMAT_RAW)
                image_response = self._image_client.get_image([img_req])
                width = image_response[0].shot.image.cols
                height = image_response[0].shot.image.rows
                image_grey = np.array(
                    Image.frombytes('P', (int(width), int(height)),
                                    data=image_response[0].shot.image.data, decoder_name='raw'))
                image_grey = self.rotate_image(image_grey, source_name)
                images[source_name] = image_grey
            except Exception as e:
                LOGGER.error(f"Failed to capture image from {source_name}: {e}")
        save_dir = input("\nEnter the directory path to save images (will be created if it doesn't exist): ").strip()
        if not save_dir:
            save_dir = os.getcwd()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for source_name, image in images.items():
            out_path = os.path.join(save_dir, f"{source_name}.png")
            cv2.imwrite(out_path, image)
            print(f" Saved {source_name} to {out_path}")
        print(" All images saved!\n")

    def image_to_bounding_box(self):
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
                self._camera_tform_body = get_a_tform_b(image_response[0].shot.transforms_snapshot,
                                                        image_response[0].shot.frame_name_image_sensor,
                                                        BODY_FRAME_NAME)
                self._body_tform_world = get_a_tform_b(image_response[0].shot.transforms_snapshot,
                                                       BODY_FRAME_NAME, VISION_FRAME_NAME)
                self._intrinsics = image_response[0].source.pinhole.intrinsics
                width = image_response[0].shot.image.cols
                height = image_response[0].shot.image.rows
                bboxes, tag_ids = self.detect_fiducial_in_image(image_response[0].shot.image, (width, height),
                                                                source_name)
                if bboxes:
                    return bboxes, tag_ids, source_name
                else:
                    self._tag_not_located = True
                    LOGGER.info(f'Failed to find bounding box for {source_name}')
            except Exception as e:
                LOGGER.error(f"Error processing image from {source_name}: {e}")
        return [], [], None

    def detect_fiducial_in_image(self, image, dim, source_name):
        try:
            image_grey = np.array(
                Image.frombytes('P', (int(dim[0]), int(dim[1])), data=image.data, decoder_name='raw'))
            image_grey = self.rotate_image(image_grey, source_name)
            detector = apriltag(families='tag36h11')
            detections = detector.detect(image_grey)
            bboxes = []
            tag_ids = []
            for det in detections:
                bbox = det.corners
                tag_id = det.tag_id
                bboxes.append(bbox)
                tag_ids.append(tag_id)
                cv2.polylines(image_grey, [np.int32(bbox)], True, (0, 0, 0), 2)
            self._image[source_name] = image_grey
            return bboxes, tag_ids
        except Exception as e:
            LOGGER.error(f"AprilTag detection failed for {source_name}: {e}")
            return [], []

    def bbox_to_image_object_pts(self, bbox):
        fiducial_height_and_width = 146
        obj_pts = np.array([[0, 0], [fiducial_height_and_width, 0], [0, fiducial_height_and_width],
                            [fiducial_height_and_width, fiducial_height_and_width]], dtype=np.float32)
        obj_points = np.insert(obj_pts, 2, 0, axis=1)
        img_pts = np.array([[bbox[3][0], bbox[3][1]], [bbox[2][0], bbox[2][1]],
                            [bbox[0][0], bbox[0][1]], [bbox[1][0], bbox[1][1]]], dtype=np.float32)
        return obj_points, img_pts

    def compute_fiducial_in_world_frame(self, tvec):
        fiducial_rt_camera_frame = np.array(
            [float(tvec[0][0]) / 1000.0,
             float(tvec[1][0]) / 1000.0,
             float(tvec[2][0]) / 1000.0])
        body_tform_fiducial = (self._camera_tform_body.inverse()).transform_point(
            fiducial_rt_camera_frame[0], fiducial_rt_camera_frame[1], fiducial_rt_camera_frame[2])
        fiducial_rt_world = self._body_tform_world.inverse().transform_point(
            body_tform_fiducial[0], body_tform_fiducial[1], body_tform_fiducial[2])
        return fiducial_rt_world

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
            self._robot_command_client.robot_command(lease=None, command=tag_cmd,
                                                     end_time_secs=time.time() + end_time)
            start_time = time.time()
            current_time = time.time()
            while (not self.final_state() and current_time - start_time < end_time):
                time.sleep(.25)
                current_time = time.time()
        return

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
        robot_to_object_ewrt_world = np.array(
            [object_rt_world.x - robot_rt_world.x, object_rt_world.y - robot_rt_world.y, 0])
        robot_to_object_ewrt_world_norm = robot_to_object_ewrt_world / np.linalg.norm(
            robot_to_object_ewrt_world)
        heading = self.get_desired_angle(robot_to_object_ewrt_world_norm)
        goto_rt_world = np.array([
            object_rt_world.x - robot_to_object_ewrt_world_norm[0] * dist_margin,
            object_rt_world.y - robot_to_object_ewrt_world_norm[1] * dist_margin
        ])
        return goto_rt_world, heading

    def set_mobility_params(self):
        obstacles = spot_command_pb2.ObstacleParams(disable_vision_body_obstacle_avoidance=True,
                                                    disable_vision_foot_obstacle_avoidance=True,
                                                    disable_vision_foot_constraint_avoidance=True,
                                                    obstacle_avoidance_padding=.001)
        body_control = self.set_default_body_control()
        if self._limit_speed:
            speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
                linear=Vec2(x=self._max_x_vel, y=self._max_y_vel), angular=self._max_ang_vel))
            if not self._avoid_obstacles:
                mobility_params = spot_command_pb2.MobilityParams(
                    obstacle_params=obstacles, vel_limit=speed_limit, body_control=body_control,
                    locomotion_hint=spot_command_pb2.HINT_AUTO)
            else:
                mobility_params = spot_command_pb2.MobilityParams(
                    vel_limit=speed_limit, body_control=body_control,
                    locomotion_hint=spot_command_pb2.HINT_AUTO)
        elif not self._avoid_obstacles:
            mobility_params = spot_command_pb2.MobilityParams(
                obstacle_params=obstacles, body_control=body_control,
                locomotion_hint=spot_command_pb2.HINT_AUTO)
        else:
            mobility_params = None
        return mobility_params

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
    def rotate_image(image, source_name):
        code = ROT_MAP.get(source_name)
        return cv2.rotate(image, code) if code is not None else image

    @staticmethod
    def make_camera_matrix(ints):
        camera_matrix = np.array([[ints.focal_length.x, ints.skew.x, ints.principal_point.x],
                                  [ints.skew.y, ints.focal_length.y, ints.principal_point.y],
                                  [0, 0, 1]])
        return camera_matrix

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
                if ch == '\x1b':
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
                if   ch.lower() == 'w': self.f.move_forward()
                elif ch.lower() == 's': self.f.move_backward()
                elif ch.lower() == 'a': self.f.strafe_left()
                elif ch.lower() == 'd': self.f.strafe_right()
                elif ch.lower() == 'q': self.f.turn_left()
                elif ch.lower() == 'e': self.f.turn_right()
                elif ch.lower() == 'r':
                    print("Search mode: rotating scan enabled.")
                    self.f.mode = 'search'
        finally:
            if not _WIN:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

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
        self._thread = threading.Thread(target=self.update)
        self._thread.start()
        return self

    def update(self):
        while self._started:
            images = self.get_image()
            for i, image in enumerate(images):
                if image.size != 0:
                    original_height, original_width = image.shape[:2]
                    resized_image = cv2.resize(
                        image, (int(original_width * .5), int(original_height * .5)),
                        interpolation=cv2.INTER_NEAREST)
                    cv2.imshow(self._sources[i], resized_image)
                    cv2.moveWindow(self._sources[i],
                                   max(int(i * original_width * .5), int(i * original_height * .5)),
                                   0)
                    cv2.waitKey(1)

    def stop(self):
        self._started = False
        cv2.destroyAllWindows()

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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--distance-margin', default=.25,
                        help='Distance [meters] that the robot should stop from the AprilTag.')
    parser.add_argument('--limit-speed', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='If the robot should limit its maximum speed.')
    parser.add_argument('--avoid-obstacles', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='If the robot should have obstacle avoidance enabled.')
    parser.add_argument('--show-preview', action='store_true', default=False,
                        help='Show camera preview windows (default: False)')
    parser.add_argument('--vel-speed', type=float, default=VELOCITY_BASE_SPEED)
    parser.add_argument('--vel-ang', type=float, default=VELOCITY_BASE_ANGULAR)
    parser.add_argument('--vel-duration', type=float, default=VELOCITY_CMD_DURATION)
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
            fiducial_follower = FollowFiducial(robot, options)
            kb = KeyboardController(fiducial_follower)
            kb.start()
            time.sleep(.1)
            if str.lower(sys.platform) != 'darwin' and options.show_preview:
                image_viewer = DisplayImagesAsync(fiducial_follower)
                image_viewer.start()
            lease_client = robot.ensure_client(LeaseClient.default_service_name)
            with bosdyn.client.lease.LeaseKeepAlive(
                lease_client, must_acquire=True, return_at_exit=True
            ):
                fiducial_follower.start()
    except RpcError as err:
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
    return False

if __name__ == '__main__':
    if not main():
        sys.exit(1)