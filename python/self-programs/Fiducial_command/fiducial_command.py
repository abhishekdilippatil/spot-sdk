# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.

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
#for wasd
VELOCITY_BASE_SPEED   = 0.5   # m/s
VELOCITY_BASE_ANGULAR = 0.8   # rad/s
VELOCITY_CMD_DURATION = 0.6   # seconds


# Use this length to make sure we're commanding the head of the robot
# to a position instead of the center.
BODY_LENGTH = 1.1

ROT_MAP = {
    'back_fisheye_image':  cv2.ROTATE_90_CLOCKWISE,
    'frontleft_fisheye_image': cv2.ROTATE_180,
    'frontright_fisheye_image': cv2.ROTATE_180,
    'left_fisheye_image':  cv2.ROTATE_90_COUNTERCLOCKWISE,
    'right_fisheye_image': cv2.ROTATE_90_CLOCKWISE,
}

class FollowFiducial(object):
    """ Detect and follow a AprilTag. """

    def __init__(self, robot, options):
        # Robot instance variable.
        self.mode = 'manual'           
        #for manual control
        self._pending_detections = None
        self._last_detected_tag_ids = []
        self._last_chosen_tag_id = None
        self._robot = robot
        self._robot_id = robot.ensure_client(RobotIdClient.default_service_name).get_id(timeout=0.4)
        self._power_client = robot.ensure_client(PowerClient.default_service_name)
        self._image_client = robot.ensure_client(ImageClient.default_service_name)
        self._robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)

        # Stopping Distance (x,y) offset from the tag and angle offset from desired angle.
        self._tag_offset = float(options.distance_margin) + BODY_LENGTH / 2.0  # meters

        # Maximum speeds.
        self._max_x_vel = 0.2
        self._max_y_vel = 0.2
        self._max_ang_vel = 0.2

        # Indicators for movement and image displays.
        self._standup = True # Stand up the robot.
        self._movement_on = True # Let the robot walk towards the AprilTag.
        self._limit_speed = options.limit_speed # Limit the robot's walking speed.
        self._avoid_obstacles = options.avoid_obstacles # Disable obstacle avoidance.

        # Epsilon distance between robot and desired go-to point.
        self._x_eps = .05
        self._y_eps = .05
        self._angle_eps = .005

        # Indicator for if motor power is on.
        self._powered_on = False

        # Counter for the number of iterations completed.
        self._attempts = 0

        # Maximum amount of iterations before powering off the motors.
        self._max_attempts = 5

        # Camera intrinsics for the current camera source being analyzed.
        self._intrinsics = None

        # Transform from the robot's camera frame to the baselink frame.
        # It is a math_helpers.SE3Pose.
        self._camera_tform_body = None

        # Transform from the robot's baselink to the world frame.
        # It is a math_helpers.SE3Pose.
        self._body_tform_world = None

        # Latest detected AprilTag's position in the world.
        self._current_tag_world_pose = np.array([])

        # Heading angle based on the camera source which detected the AprilTag.
        self._angle_desired = None

        # Dictionary mapping camera source to it's latest image taken.
        self._image = dict()

        # List of all possible camera sources.
        self._source_names = [
            src.name for src in self._image_client.list_image_sources()
            if (src.image_type == image_pb2.ImageSource.IMAGE_TYPE_VISUAL and 'depth' not in src.name)
        ]
        print(self._source_names)

        # Dictionary mapping camera source to previously computed extrinsics.
        self._camera_to_extrinsics_guess = self.populate_source_dict()

        # Camera source which a bounding box was last detected in.
        self._previous_source = None

    @property
    def robot_state(self):
        """Get latest robot state proto."""
        return self._robot_state_client.get_robot_state()

    @property
    def image(self):
        """Return the current image associated with each source name."""
        return self._image

    @property
    def image_sources_list(self):
        """Return the list of camera sources."""
        return self._source_names

    def populate_source_dict(self):
        """Fills dictionary of the most recently computed camera extrinsics with the camera source.
           The initial boolean indicates if the extrinsics guess should be used."""
        camera_to_extrinsics_guess = dict()
        for src in self._source_names:
            # Dictionary values: use_extrinsics_guess bool, (rotation vector, translation vector) tuple.
            camera_to_extrinsics_guess[src] = (False, (None, None))
        return camera_to_extrinsics_guess

    def start(self):
        """Claim lease of robot and start the AprilTag follower."""
        self._robot.time_sync.wait_for_sync()
        # Stand the robot up.
        if self._standup:
            self.power_on()
            blocking_stand(self._robot_command_client)

            # Delay grabbing image until spot is standing (or close enough to upright).
            time.sleep(.35)

        while self._attempts <= self._max_attempts:
            bboxes, tag_ids, source_name = self.image_to_bounding_box()
            if bboxes and tag_ids:
                self._previous_source = source_name
                self.on_tags_detected(bboxes, tag_ids, source_name)
            else:
                # Only auto-scan if we're not in manual driving (e.g., after user asked to search)
                if self.mode != 'manual':
                    print("No AprilTags found. Rotating to scan...")
                    self.sweep_yaw()
                    self._attempts += 1

            # Optional: a tiny sleep so the keyboard thread gets CPU time.
            time.sleep(0.02)


            #     current_tag_ids = sorted(tag_ids)
            #     # Prompt for tag selection if new set or if last selection not visible
            #     if (self._last_chosen_tag_id is None or 
            #         self._last_chosen_tag_id not in tag_ids or
            #         current_tag_ids != self._last_detected_tag_ids):

            #         # Show distances (optional helper)
            #         self.print_detected_tags_with_distance(bboxes, tag_ids)
            #         # print("\nDetected AprilTags in view:")
            #         # for idx, tag_id in enumerate(tag_ids):
            #         #     obj_points, img_points = self.bbox_to_image_object_pts(bboxes[idx])
            #         #     camera = self.make_camera_matrix(self._intrinsics)
            #         #     _, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera, np.zeros((5, 1)))
            #         #     dist_m = math.sqrt(float(tvec[0][0])**2 + float(tvec[1][0])**2 + float(tvec[2][0])**2) / 1000.0
            #         #     print(f"  Tag {idx+1}: ID={tag_id}, Distance={dist_m:.2f} m")
                    
            #         # Ask ONLY the two questions here
            #         action, chosen_tag_id = self.prompt_follow_or_rotate(tag_ids)
            #         if action == "rotate":
            #             print("Rotating 90 degrees as requested.")
            #             self._previous_source = None
            #             self.sweep_yaw()
            #             self._last_chosen_tag_id = None
            #             self._last_detected_tag_ids = []
            #             continue  # go back to scanning loop

            #         # action == "follow"
            #         self._last_chosen_tag_id = chosen_tag_id
            #         self._last_detected_tag_ids = current_tag_ids

            #     if self._last_chosen_tag_id in tag_ids:
            #         chosen_index = tag_ids.index(self._last_chosen_tag_id)
            #         obj_points, img_points = self.bbox_to_image_object_pts(bboxes[chosen_index])
            #         camera = self.make_camera_matrix(self._intrinsics)
            #         _, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera, np.zeros((5, 1)))
            #         vision_tform_fiducial_position = self.compute_fiducial_in_world_frame(tvec)
            #         fiducial_rt_world = geometry_pb2.Vec3(
            #             x=vision_tform_fiducial_position[0],
            #             y=vision_tform_fiducial_position[1],
            #             z=vision_tform_fiducial_position[2]
            #         )
            #         print(f"\n Spot is walking to AprilTag ID {self._last_chosen_tag_id}...\n")
            #         self.go_to_tag(fiducial_rt_world)
            #         print(f"\n Spot reached tag ID {self._last_chosen_tag_id}.")
            #         next_action = self.prompt_next_action()
            #         if next_action == "q":
            #             print("Quitting as requested by user.")
            #             break
            #         elif next_action == "r":
            #             print("Rotating 90 degrees as requested by user.")
            #             self._previous_source = None
            #             self.sweep_yaw()
            #         elif next_action == "s":
            #             print("Staying here.")
            #             self._last_chosen_tag_id = None
            #             self._last_detected_tag_ids = []
            #         elif next_action == "n":
            #             print("Searching for new tags.")
            #             self._previous_source = None
            #             self._last_chosen_tag_id = None
            #             self._last_detected_tag_ids = []
            # else:
            #     print("No AprilTags found. Rotating to scan...")
            #     # Escalate the yaw sweep each time, then stop after 5 tries
            #     self._previous_source = None
            #     self._attempts += 1  # increment attempts at finding an AprilTag

            # if self._attempts > self._max_attempts:
            #         print(f"No AprilTags found after {self._max_attempts} scans. Terminating.")
            #         break

            #     # Escalating sweep: 90°, 180°, 270°, 360°, 360°
            #     yaw_deg = min(90 * self._attempts, 360)
            #     # More steps for wider sweeps; clamp to something reasonable
            #     steps = min(9 + 4 * self._attempts, 45)
            #     # Slightly faster pauses as we escalate, but keep ≥ 0.4s
            #     pause = max(0.4, 1.0 - 0.1 * self._attempts)

            #     print(f"No AprilTags found. Sweeping yaw across ±{yaw_deg/2:.0f}° "
            #         f"with {steps} steps (attempt {self._attempts}/{self._max_attempts})...")
            #     self.sweep_yaw(yaw_range=math.radians(yaw_deg), steps=steps, pause=pause)

    
            #     # Uncomment the line below if you want this extra motion:
            #     self.rotate_in_place(angle_rad=math.radians(90), angular_speed=0.8)

        # Power off at the conclusion of the example.
        if self._powered_on:
            self.power_off()

    def power_on(self):
        """Power on the robot."""
        self._robot.power_on()
        self._powered_on = True
        print(f'Powered On {self._robot.is_powered_on()}')

    def power_off(self):
        """Power off the robot."""
        self._robot.power_off()
        print(f'Powered Off {not self._robot.is_powered_on()}')

        # ---- WASD-style velocity helper & atomic moves ----
    def _velocity_command(self, desc='', v_x=0.0, v_y=0.0, v_rot=0.0, duration=VELOCITY_CMD_DURATION):
        if not (self._movement_on and self._powered_on):
            print(f"Cannot {desc} — movement disabled or motors off.")
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
        # Called from the scan loop when we first see tags in manual mode.
        ids = [i['id'] for i in infos]
        closest = min(infos, key=lambda x: x['dist'])
        print("\n[AprilTag] Detected IDs:", ids)
        print(f"Closest: ID={closest['id']} at ~{closest['dist']:.2f} m.")
        print("Press [F] to FOLLOW closest, or [C] to CONTINUE manual driving.")

    def _pnp_each(self, bboxes, tag_ids):
        """Compute PnP for each detection; return [{'id', 'tvec', 'dist'}]."""
        out = []
        camera = self.make_camera_matrix(self._intrinsics)
        for idx, tag_id in enumerate(tag_ids):
            obj_points, img_points = self.bbox_to_image_object_pts(bboxes[idx])
            ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera, np.zeros((5, 1)))
            if not ok:
                continue
            dist_m = math.sqrt(float(tvec[0][0])**2 + float(tvec[1][0])**2 + float(tvec[2][0])**2) / 1000.0
            out.append({'id': int(tag_id), 'tvec': tvec, 'dist': dist_m})
        return out

    def on_tags_detected(self, bboxes, tag_ids, source_name):
        """Called by the scan loop whenever tags are visible."""
        if self.mode != 'manual':
            return  # ignore while we're already prompting or following
        infos = self._pnp_each(bboxes, tag_ids)
        if not infos:
            return
        self._pending_detections = infos
        self.mode = 'prompt'
        self._announce_tag_prompt(infos)

    def follow_detected_closest(self):
        """Follow the closest of the last detected tags."""
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
        """Abort an ongoing follow and go back to manual (mapped to [M])."""
        try:
            self._robot_command_client.robot_command(command=RobotCommandBuilder.stop_command())
        finally:
            self.mode = 'manual'
            print("Follow cancelled. Manual mode.")


    def prompt_next_action(self):
        """Prompt the user for the next action."""
        print("\nSpot has reached the tag.")
        print("What should Spot do next?")
        print("  [p] Take photo from all cameras and save")
        print("  [n] Find another tag and follow")
        print("  [r] Rotate 90 degrees and search")
        print("  [s] Stay here")
        print("  [q] Quit")
        while True:
            action = input("Enter your choice [n/q/r/s/p]: ").strip().lower()
            if action in ["n", "q", "r", "s", "p"]:
                if action == "p":
                    self.capture_and_save_photos_from_all_cameras()
                    continue #Show menu after taking photo
                return action
            else:
                print("Invalid input. Please enter one of: n, q, r, s, p.")
    def prompt_follow_or_rotate(self, tag_ids):
        """Ask whether to follow a visible tag or rotate to search for another."""
        while True:
            choice = input("\nDetected tags. Follow a tag [f] or rotate 90° to find another [r]? ").strip().lower()
            if choice == 'f':
                # If there's only one tag, follow it. Otherwise ask which ID.
                if len(tag_ids) == 1:
                    return ("follow", tag_ids[0])
                else:
                    while True:
                        try:
                            user_input = input(f"Enter the ID of the tag to follow from {tag_ids}: ").strip()
                            chosen_tag_id = int(user_input)
                            if chosen_tag_id in tag_ids:
                                return ("follow", chosen_tag_id)
                            print(f"Tag ID {chosen_tag_id} is not in {tag_ids}. Try again.")
                        except Exception:
                            print("Invalid input. Please enter a valid numeric tag ID.")
            elif choice == 'r':
                return ("rotate", None)
            else:
                print("Please enter 'f' to follow or 'r' to rotate.")

    def print_detected_tags_with_distance(self, bboxes, tag_ids):
        """Compute and print distances for currently detected tags."""
        camera = self.make_camera_matrix(self._intrinsics)
        print("\nDetected AprilTags in view:")
        for idx, tag_id in enumerate(tag_ids):
            obj_points, img_points = self.bbox_to_image_object_pts(bboxes[idx])
            _, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera, np.zeros((5, 1)))
            dist_m = math.sqrt(float(tvec[0][0])**2 + float(tvec[1][0])**2 + float(tvec[2][0])**2) / 1000.0
            print(f"  Tag {idx+1}: ID={tag_id}, Distance={dist_m:.2f} m")


    def rotate_in_place(self, angle_rad=math.pi/2, angular_speed=1): # angle_rad in radians, angle can be changed by changing pi/2
        """Rotate in place."""
        print(f" Rotating {round(math.degrees(angle_rad))} degrees to scan...")
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
                print(" Spot reports stop command expired. (Usually safe to ignore)", e)

    def sweep_yaw(self, yaw_range=math.radians(90), steps=9, pause=1.0):
        """Sweep Spot's body yaw left and right to look for AprilTags."""
        self._previous_source = None # Next image will not pass to last camera source
        max_yaw = yaw_range / 2
        yaws = np.linspace(-max_yaw, max_yaw, steps)
        for yaw in yaws:
            orientation = EulerZXY(yaw, 0.0, 0.0)
            stand_cmd = RobotCommandBuilder.synchro_stand_command(body_height=0.0, footprint_R_body=orientation)
            self._robot_command_client.robot_command(lease=None, command=stand_cmd, end_time_secs=time.time() + pause)
            time.sleep(pause)
            self._last_chosen_tag_id = None
            self._last_detected_tag_ids = []
    
    def capture_and_save_photos_from_all_cameras(self):
        """
        Prompts the user to capture a photo from all cameras and asks where to save.
        """
        print("\n Capture photos from all cameras")
        input("Press Enter to take photos from all cameras...")

        images = {}
        for source_name in self._source_names:
            print(f" Capturing image from {source_name}...")
            img_req = build_image_request(source_name, quality_percent=100,
                                        image_format=image_pb2.Image.FORMAT_RAW)
            image_response = self._image_client.get_image([img_req])
            width = image_response[0].shot.image.cols
            height = image_response[0].shot.image.rows

            # Convert raw bytes to 8-bit greyscale numpy array
            image_grey = np.array(
                Image.frombytes('P', (int(width), int(height)),
                                data=image_response[0].shot.image.data, decoder_name='raw'))

            image_grey = self.rotate_image(image_grey, source_name)
            images[source_name] = image_grey

        # Ask user for directory
        save_dir = input("\nEnter the directory path to save images (will be created if it doesn't exist): ").strip()
        if not save_dir:
            save_dir = os.getcwd()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save all images
        for source_name, image in images.items():
            out_path = os.path.join(save_dir, f"{source_name}.png")
            cv2.imwrite(out_path, image)
            print(f" Saved {source_name} to {out_path}")
        print(" All images saved!\n")

    def image_to_bounding_box(self):
        """Determine which camera source has a AprilTag..
           Return the bounding box of the first detected AprilTag."""
        #Iterate through all five camera sources to check for a AprilTag.
        for i in range(len(self._source_names) + 1):
            # Get the image from the source camera.
            if i == 0:
                if self._previous_source is not None:
                    # Prioritize the camera the AprilTag was last detected in.
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
            
            # Camera intrinsics for the given source camera.
            self._intrinsics = image_response[0].source.pinhole.intrinsics
            width = image_response[0].shot.image.cols
            height = image_response[0].shot.image.rows

            # detect given AprilTag in image and return the bounding box of it
            bboxes, tag_ids = self.detect_fiducial_in_image(image_response[0].shot.image, (width, height),
                                                   source_name)
            if bboxes:
                return bboxes, tag_ids, source_name
            else:
                self._tag_not_located = True
                print(f'Failed to find bounding box for {source_name}')
        return [],[], None

    def detect_fiducial_in_image(self, image, dim, source_name):
        """Detect the AprilTag within a single image and return its bounding box."""
        image_grey = np.array(
        Image.frombytes('P', (int(dim[0]), int(dim[1])), data=image.data, decoder_name='raw'))

        #Rotate each image such that it is upright
        image_grey = self.rotate_image(image_grey, source_name)

        #Make the image greyscale to use bounding box detections
        detector = apriltag(families='tag36h11')
        detections = detector.detect(image_grey)

        bboxes = []
        tag_ids = []
        for det in detections:
            # Draw the bounding box detection in the image.
            bbox = det.corners
            tag_id = det.tag_id
            bboxes.append(bbox)
            tag_ids.append(tag_id)
            cv2.polylines(image_grey, [np.int32(bbox)], True, (0, 0, 0), 2)
        self._image[source_name] = image_grey
        return bboxes, tag_ids

    def bbox_to_image_object_pts(self, bbox):
        """Determine the object points and image points for the bounding box.
           The origin in object coordinates = top left corner of the AprilTag.
           Order both points sets following: (TL,TR, BL, BR)"""
        fiducial_height_and_width = 146  # mm
        obj_pts = np.array([[0, 0], [fiducial_height_and_width, 0], [0, fiducial_height_and_width],
                            [fiducial_height_and_width, fiducial_height_and_width]],
                           dtype=np.float32)
        #insert a 0 as the third coordinate (xyz)
        obj_points = np.insert(obj_pts, 2, 0, axis=1)

        #['lb-rb-rt-lt']
        img_pts = np.array([[bbox[3][0], bbox[3][1]], [bbox[2][0], bbox[2][1]],
                            [bbox[0][0], bbox[0][1]], [bbox[1][0], bbox[1][1]]], dtype=np.float32)
        return obj_points, img_pts

    def compute_fiducial_in_world_frame(self, tvec):
        """Transform the tag position from camera coordinates to world coordinates."""
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
        """Use the position of the april tag in vision world frame and command the robot."""
        # Compute the go-to point (offset by .5m from the AprilTag position) and the heading at
        # this point.
        self._current_tag_world_pose, self._angle_desired = self.offset_tag_pose(
            fiducial_rt_world, self._tag_offset)
        #Command the robot to go to the tag in kinematic odometry frame
        mobility_params = self.set_mobility_params()
        tag_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=self._current_tag_world_pose[0], goal_y=self._current_tag_world_pose[1],
            goal_heading=self._angle_desired, frame_name=VISION_FRAME_NAME, params=mobility_params,
            body_height=0.0, locomotion_hint=spot_command_pb2.HINT_AUTO)
        end_time = 30.0
        if self._movement_on and self._powered_on:
            #Issue the command to the robot
            self._robot_command_client.robot_command(lease=None, command=tag_cmd,
                                                     end_time_secs=time.time() + end_time)
            #Feedback to check and wait until the robot is in the desired position or timeout
            start_time = time.time()
            current_time = time.time()
            while (not self.final_state() and current_time - start_time < end_time):
                time.sleep(.25)
                current_time = time.time()
        return

    def final_state(self):
        """Check if the current robot state is within range of the AprilTags position."""
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
        """Compute heading based on the vector from robot to object."""
        zhat = [0.0, 0.0, 1.0]
        yhat = np.cross(zhat, xhat)
        mat = np.array([xhat, yhat, zhat]).transpose()
        return Quat.from_matrix(mat).to_yaw()

    def offset_tag_pose(self, object_rt_world, dist_margin=1.0):
        """Offset the go-to location of the AprilTag and compute the desired heading."""
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
        """Set robot mobility params to disable obstacle avoidance."""
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
            #When set to none, RobotCommandBuilder populates with good default values
            mobility_params = None
        return mobility_params

    @staticmethod
    def set_default_body_control():
        """Set default body control params to current body position"""
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
        """Transform the ImageResponse proto intrinsics into a camera matrix."""
        camera_matrix = np.array([[ints.focal_length.x, ints.skew.x, ints.principal_point.x],
                                  [ints.skew.y, ints.focal_length.y, ints.principal_point.y],
                                  [0, 0, 1]])
        return camera_matrix
    
class KeyboardController(threading.Thread):
    """Non-blocking console keyboard for WASD + [F]/[C] prompt + [M] cancel."""
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

                # Global escape hatch
                if ch == '\x1b':  # ESC
                    self.stop()
                    continue

                # If we're prompting: [F] follow, [C] continue
                if self.f.mode == 'prompt':
                    if ch.lower() == 'f':
                        self.f.follow_detected_closest()
                    elif ch.lower() == 'c':
                        self.f.dismiss_prompt()
                    continue

                # If following: allow cancel with [M]
                if self.f.mode == 'following':
                    if ch.lower() == 'm':
                        self.f.cancel_follow()
                    continue

                # Manual driving keys (nudges from wasd.py)
                if   ch.lower() == 'w': self.f.move_forward()
                elif ch.lower() == 's': self.f.move_backward()
                elif ch.lower() == 'a': self.f.strafe_left()
                elif ch.lower() == 'd': self.f.strafe_right()
                elif ch.lower() == 'q': self.f.turn_left()
                elif ch.lower() == 'e': self.f.turn_right()
                # Optional: 'r' to enter search mode
                elif ch.lower() == 'r':
                    print("Search mode: rotating scan enabled.")
                    self.f.mode = 'search'
        finally:
            if not _WIN:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)



class DisplayImagesAsync(object):
    """Display the images Spot sees from all five cameras."""
    def __init__(self, fiducial_follower):
        self._fiducial_follower = fiducial_follower
        self._thread = None
        self._started = False
        self._sources = []

    def get_image(self):
        """Retrieve current images (with bounding boxes) from the AprilTag detector."""
        images = self._fiducial_follower.image
        image_by_source = []
        for s_name in self._sources:
            if s_name in images:
                image_by_source.append(images[s_name])
            else:
                image_by_source.append(np.array([]))
        return image_by_source

    def start(self):
        """Initialize the thread to display the images."""
        if self._started:
            return None
        self._sources = self._fiducial_follower.image_sources_list
        self._started = True
        self._thread = threading.Thread(target=self.update)
        self._thread.start()
        return self

    def update(self):
        """Update the images being displayed to match that seen by the robot."""
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
        """Stop the thread and the image displays."""
        self._started = False
        cv2.destroyAllWindows()


class Exit(object):
    """Handle exiting on SIGTERM."""
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
        """Return if sigterm received and program should end."""
        return self._kill_now

def main():
    """Command-line interface."""
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

    # (Optional) if you added WASD CLI knobs in step 4, include them here too:
    parser.add_argument('--vel-speed', type=float, default=VELOCITY_BASE_SPEED)
    parser.add_argument('--vel-ang', type=float, default=VELOCITY_BASE_ANGULAR)
    parser.add_argument('--vel-duration', type=float, default=VELOCITY_CMD_DURATION)

    options = parser.parse_args()

    sdk = create_standard_sdk('FollowFiducialClient')
    robot = sdk.create_robot(options.hostname)

    fiducial_follower = None
    image_viewer = None
    kb = None  # <— keyboard thread handle

    try:
        with Exit():
            bosdyn.client.util.authenticate(robot)
            robot.start_time_sync()

            assert not robot.is_estopped(), 'Robot is estopped. Use E-Stop client to configure.'

            # Build controller & follower
            fiducial_follower = FollowFiducial(robot, options)

            # --- START KEYBOARD THREAD HERE ---
            # This enables non-blocking WASD and [F]/[C] prompt handling while the main loop runs.
            kb = KeyboardController(fiducial_follower)
            kb.start()
            # -----------------------------------

            time.sleep(.1)

            if str.lower(sys.platform) != 'darwin' and options.show_preview:
                # Display the detected bounding boxes on the images.
                image_viewer = DisplayImagesAsync(fiducial_follower)
                image_viewer.start()

            lease_client = robot.ensure_client(LeaseClient.default_service_name)
            with bosdyn.client.lease.LeaseKeepAlive(
                lease_client, must_acquire=True, return_at_exit=True
            ):
                # This is your existing loop (now non-blocking thanks to the keyboard thread).
                fiducial_follower.start()

    except RpcError as err:
        LOGGER.error('Failed to communicate with robot: %s', err)

    finally:
        # Close preview windows if open
        if image_viewer is not None:
            image_viewer.stop()

        # --- STOP KEYBOARD THREAD CLEANLY ---
        if kb is not None:
            kb.stop()
            try:
                kb.join(timeout=1.0)
            except Exception:
                pass
        # ------------------------------------

    return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)

# def main():
#     """Command-line interface."""
#     import argparse

#     parser = argparse.ArgumentParser()
#     bosdyn.client.util.add_base_arguments(parser)
#     parser.add_argument('--distance-margin', default=.25, ##default value by boston dynamic was 0.5
#                         help='Distance [meters] that the robot should stop from the AprilTag.')
#     parser.add_argument('--limit-speed', default=True, type=lambda x: (str(x).lower() == 'true'),
#                         help='If the robot should limit its maximum speed.')
#     parser.add_argument('--avoid-obstacles', default=True, type=lambda x:
#                         (str(x).lower() == 'true'),
#                         help='If the robot should have obstacle avoidance enabled.')
#     parser.add_argument('--show-preview', action='store_true', default=False,
#                     help='Show camera preview windows (default: False)')
#     options = parser.parse_args()

#     sdk = create_standard_sdk('FollowFiducialClient')
#     robot = sdk.create_robot(options.hostname)

#     fiducial_follower = None
#     image_viewer = None
#     try:
#         with Exit():
#             bosdyn.client.util.authenticate(robot)
#             robot.start_time_sync()

#             assert not robot.is_estopped(), 'Robot is estopped. Use E-Stop client to configure.'

#             fiducial_follower = FollowFiducial(robot, options)
#             time.sleep(.1)
#             if str.lower(sys.platform) != 'darwin' and options.show_preview:
#                 # Display the detected bounding boxes on the images when using the april tag library.
#                 image_viewer = DisplayImagesAsync(fiducial_follower)
#                 image_viewer.start()
#             lease_client = robot.ensure_client(LeaseClient.default_service_name)
#             with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True,
#                                                     return_at_exit=True):
#                 fiducial_follower.start()
#     except RpcError as err:
#         LOGGER.error('Failed to communicate with robot: %s', err)
#     finally:
#         if image_viewer is not None:
#             image_viewer.stop()

#     return False

# if __name__ == '__main__':
#     if not main():
#         sys.exit(1)
