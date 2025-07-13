# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.

""" Detect and follow fiducial tags (AprilTag-only version). """
import logging
import math
import signal
import sys
import threading
import time
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

LOGGER = logging.getLogger()

BODY_LENGTH = 1.1

class FollowFiducial(object):
    """ Detect and follow a fiducial with Spot (AprilTag only). """

    def __init__(self, robot, options):
        self._last_detected_tag_ids = []
        self._last_chosen_tag_id = None
        self._robot = robot
        self._robot_id = robot.ensure_client(RobotIdClient.default_service_name).get_id(timeout=0.4)
        self._power_client = robot.ensure_client(PowerClient.default_service_name)
        self._image_client = robot.ensure_client(ImageClient.default_service_name)
        self._robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)

        self._tag_offset = float(options.distance_margin) + BODY_LENGTH / 2.0  # meters

        self._max_x_vel = 0.2
        self._max_y_vel = 0.2
        self._max_ang_vel = 0.2

        self._standup = True
        self._movement_on = True
        self._limit_speed = options.limit_speed
        self._avoid_obstacles = options.avoid_obstacles

        self._x_eps = .05
        self._y_eps = .05
        self._angle_eps = .075

        self._powered_on = False
        self._attempts = 0
        self._max_attempts = 100000

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
        print(self._source_names)
        self._camera_to_extrinsics_guess = self.populate_source_dict()
        self._previous_source = None

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

    def start(self):
        """Claim lease of robot and start the fiducial follower (AprilTags only)."""
        self._robot.time_sync.wait_for_sync()
        if self._standup:
            self.power_on()
            blocking_stand(self._robot_command_client)
            time.sleep(.35)

        while self._attempts <= self._max_attempts:
            bboxes, tag_ids, source_name = self.image_to_bounding_box()
            if bboxes and tag_ids:
                self._previous_source = source_name
                current_tag_ids = sorted(tag_ids)
                # Prompt for tag selection if new set or if last selection not visible
                if (self._last_chosen_tag_id is None or 
                    self._last_chosen_tag_id not in tag_ids or
                    current_tag_ids != self._last_detected_tag_ids):
                    print("\nDetected AprilTags in view:")
                    for idx, tag_id in enumerate(tag_ids):
                        obj_points, img_points = self.bbox_to_image_object_pts(bboxes[idx])
                        camera = self.make_camera_matrix(self._intrinsics)
                        _, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera, np.zeros((5, 1)))
                        dist_m = math.sqrt(float(tvec[0][0])**2 + float(tvec[1][0])**2 + float(tvec[2][0])**2) / 1000.0
                        print(f"  Tag {idx+1}: ID={tag_id}, Distance={dist_m:.2f} m")
                    self._last_chosen_tag_id = None
                    while self._last_chosen_tag_id is None:
                        try:
                            user_input = input(f"\nEnter the ID of the tag you want Spot to follow (from {tag_ids}): ")
                            chosen_tag_id = int(user_input)
                            if chosen_tag_id not in tag_ids:
                                print(f"Tag ID {chosen_tag_id} is not detected. Try again.")
                            else:
                                self._last_chosen_tag_id = chosen_tag_id
                        except Exception:
                            print("Invalid input. Please enter a valid tag ID.")
                    self._last_detected_tag_ids = current_tag_ids

                if self._last_chosen_tag_id in tag_ids:
                    chosen_index = tag_ids.index(self._last_chosen_tag_id)
                    obj_points, img_points = self.bbox_to_image_object_pts(bboxes[chosen_index])
                    camera = self.make_camera_matrix(self._intrinsics)
                    _, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera, np.zeros((5, 1)))
                    vision_tform_fiducial_position = self.compute_fiducial_in_world_frame(tvec)
                    fiducial_rt_world = geometry_pb2.Vec3(
                        x=vision_tform_fiducial_position[0],
                        y=vision_tform_fiducial_position[1],
                        z=vision_tform_fiducial_position[2]
                    )
                    print(f"\nSpot is walking to fiducial tag ID {self._last_chosen_tag_id}...\n")
                    self.go_to_tag(fiducial_rt_world)
                    print(f"\nSpot reached tag ID {self._last_chosen_tag_id}.")
                    next_action = self.prompt_next_action()
                    if next_action == "q":
                        print("Quitting as requested by user.")
                        break
                    elif next_action == "r":
                        print("Rotating 90 degrees as requested by user.")
                        self.rotate_in_place(angle_rad=math.pi/2, angular_speed=0.5)
                        self._last_chosen_tag_id = None
                        self._last_detected_tag_ids = []
                    elif next_action == "s":
                        print("Staying here.")
                        self._last_chosen_tag_id = None
                        self._last_detected_tag_ids = []
                    elif next_action == "n":
                        print("Searching for new tags.")
                        self._last_chosen_tag_id = None
                        self._last_detected_tag_ids = []
            else:
                print("[INFO] No fiducials found. Rotating 90 degrees to scan...")
                self.rotate_in_place(angle_rad=math.pi/2, angular_speed=0.5)
                self._attempts += 1

        if self._powered_on:
            self.power_off()

    def power_on(self):
        self._robot.power_on()
        self._powered_on = True
        print(f'Powered On {self._robot.is_powered_on()}')

    def power_off(self):
        self._robot.power_off()
        print(f'Powered Off {not self._robot.is_powered_on()}')

    def prompt_next_action(self):
        print("\n[USER] Spot has reached the tag.")
        print("What should Spot do next?")
        print("  [n] Find another tag and follow")
        print("  [q] Quit")
        print("  [r] Rotate 90 degrees and search")
        print("  [s] Stay here")
        while True:
            action = input("Enter your choice [n/q/r/s]: ").strip().lower()
            if action in ["n", "q", "r", "s"]:
                return action
            else:
                print("Invalid input. Please enter one of: n, q, r, s.")

    def rotate_in_place(self, angle_rad=math.pi/2, angular_speed=1):
        print(f"[INFO] Rotating {round(math.degrees(angle_rad))} degrees to scan...")
        mobility_params = self.set_mobility_params()
        spin_cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=0.0,
            v_y=0.0,
            v_rot=angular_speed if angle_rad >= 0 else -angular_speed,
            frame_name=BODY_FRAME_NAME,
            params=mobility_params,
            body_height=0.0,
            locomotion_hint=spot_command_pb2.HINT_AUTO
        )
        if self._movement_on and self._powered_on:
            duration = abs(angle_rad / angular_speed)
            self._robot_command_client.robot_command(
                lease=None, command=spin_cmd, end_time_secs=time.time() + duration
            )
            time.sleep(duration)
            time.sleep(1)
            stop_cmd = RobotCommandBuilder.stop_command()
            try:
                self._robot_command_client.robot_command(lease=None, command=stop_cmd)
            except bosdyn.client.robot_command.ExpiredError as e:
                print("[WARN] Spot reports stop command expired. Usually safe to ignore:", e)

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
                print(f'Failed to find bounding box for {source_name}')
        return [],[], None

    def detect_fiducial_in_image(self, image, dim, source_name):
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

    def bbox_to_image_object_pts(self, bbox):
        fiducial_height_and_width = 146  # mm
        obj_pts = np.array([[0, 0], [fiducial_height_and_width, 0], [0, fiducial_height_and_width],
                            [fiducial_height_and_width, fiducial_height_and_width]],
                           dtype=np.float32)
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
        end_time = 5.0
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
        if source_name == 'frontleft_fisheye_image':
            image = cv2.rotate(image, rotateCode=0)
        elif source_name == 'right_fisheye_image':
            image = cv2.rotate(image, rotateCode=1)
        elif source_name == 'frontright_fisheye_image':
            image = cv2.rotate(image, rotateCode=0)
        return image

    @staticmethod
    def make_camera_matrix(ints):
        camera_matrix = np.array([[ints.focal_length.x, ints.skew.x, ints.principal_point.x],
                                  [ints.skew.y, ints.focal_length.y, ints.principal_point.y],
                                  [0, 0, 1]])
        return camera_matrix


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
                        help='Distance [meters] that the robot should stop from the fiducial.')
    parser.add_argument('--limit-speed', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='If the robot should limit its maximum speed.')
    parser.add_argument('--avoid-obstacles', default=True, type=lambda x:
                        (str(x).lower() == 'true'),
                        help='If the robot should have obstacle avoidance enabled.')
    parser.add_argument('--show-preview', action='store_true', default=False,
                    help='Show camera preview windows (default: False)')
    options = parser.parse_args()

    sdk = create_standard_sdk('FollowFiducialClient')
    robot = sdk.create_robot(options.hostname)

    fiducial_follower = None
    image_viewer = None
    try:
        with Exit():
            bosdyn.client.util.authenticate(robot)
            robot.start_time_sync()

            assert not robot.is_estopped(), 'Robot is estopped. Use E-Stop client to configure.'

            fiducial_follower = FollowFiducial(robot, options)
            time.sleep(.1)
            if str.lower(sys.platform) != 'darwin' and options.show_preview:
                image_viewer = DisplayImagesAsync(fiducial_follower)
                image_viewer.start()
            lease_client = robot.ensure_client(LeaseClient.default_service_name)
            with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True,
                                                    return_at_exit=True):
                fiducial_follower.start()
    except RpcError as err:
        LOGGER.error('Failed to communicate with robot: %s', err)
    finally:
        if image_viewer is not None:
            image_viewer.stop()

    return False

if __name__ == '__main__':
    if not main():
        sys.exit(1)
