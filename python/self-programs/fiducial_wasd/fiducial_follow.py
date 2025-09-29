# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

""" Detect and follow fiducial tags. """
import logging
import math
import signal
import sys
import threading
import time
from sys import platform
import winsound

import cv2
import numpy as np
from PIL import Image

import bosdyn.client
import bosdyn.client.util
from bosdyn import geometry
from bosdyn.api import geometry_pb2, image_pb2, trajectory_pb2, world_object_pb2
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
from bosdyn.client.robot_id import RobotIdClient, version_tuple
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient

#pylint: disable=no-member
LOGGER = logging.getLogger("fiducial_follow")

# Use this length to make sure we're commanding the head of the robot
# to a position instead of the center.
BODY_LENGTH = 1.1


class FollowFiducial(object):
    """ Detect and follow a fiducial with Spot."""

    def __init__(self, robot, options):
        # Robot instance variable.
        self._robot = robot
        self._robot_id = robot.ensure_client(RobotIdClient.default_service_name).get_id(timeout=0.4)
        self._power_client = robot.ensure_client(PowerClient.default_service_name)
        self._image_client = robot.ensure_client(ImageClient.default_service_name)
        self._robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self._world_object_client = robot.ensure_client(WorldObjectClient.default_service_name)

        # Stopping Distance (x,y) offset from the tag and angle offset from desired angle.
        self._tag_offset = float(options.distance_margin) + BODY_LENGTH / 2.0  # meters

        # Maximum speeds.
        self._max_x_vel = 0.5
        self._max_y_vel = 0.5
        self._max_ang_vel = 1.0

        # Indicator if fiducial detection's should be from the world object service using
        # spot's perception system or detected with the apriltag library. If the software version
        # does not include the world object service, then default to april tag library.
        self._use_world_object_service = (options.use_world_objects and
                                          self.check_if_version_has_world_objects(self._robot_id))

        # Indicators for movement and image displays.
        self._standup = True  # Stand up the robot.
        self._movement_on = True  # Let the robot walk towards the fiducial.
        self._limit_speed = options.limit_speed  # Limit the robot's walking speed.
        self._avoid_obstacles = options.avoid_obstacles  # Disable obstacle avoidance.

        # Fiducial ID to follow.
        self._visible_ids = []
        self._selected_tag_id = None
        self._follow_selected_only = False
        self._master_tag_id = 5

        # Epsilon distance between robot and desired go-to point.
        self._x_eps = .05
        self._y_eps = .05
        self._angle_eps = .075

        # Indicator for if motor power is on.
        self._powered_on = False

        # Counter for the number of iterations completed.
        self._attempts = 0

        # Maximum amount of iterations before powering off the motors.
        self._max_attempts = 100000

        # Latest detected fiducial's position in the world.
        self._current_tag_world_pose = np.array([])

        # Heading angle based on the camera source which detected the fiducial.
        self._angle_desired = None

        # Dictionary mapping camera source to it's latest image taken.
        self._image = dict()
        
        self._stop = threading.Event()  # Event to stop the robot.

        # List of all possible camera sources.
        self._source_names = [
            src.name for src in self._image_client.list_image_sources() if
            (src.image_type == image_pb2.ImageSource.IMAGE_TYPE_VISUAL and 'depth' not in src.name)
        ]
        print(self._source_names)

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
    
    def check_if_version_has_world_objects(self, robot_id):
        """Check that software version contains world object service."""
        # World object service was released in spot-sdk version 1.2.0
        return version_tuple(robot_id.software_release.version) >= (1, 2, 0)

    def start(self):
        """Claim lease of robot and start the fiducial follower."""
        if hasattr(self, "_stop"):
            self._stop.clear()
        self._movement_on = True
        self._attempts = 0
        # Stand the robot up.
        if self._standup:
            self.power_on()
            blocking_stand(self._robot_command_client)

            # Delay grabbing image until spot is standing (or close enough to upright).
            time.sleep(.35)

        while not self._stop.is_set() and self._attempts <= self._max_attempts:
            fiducials = self.get_fiducial_objects() or []

            if fiducials:
                # Log all visible IDs
                self._visible_ids = [f.apriltag_properties.tag_id for f in fiducials]
                LOGGER.info("Visible fiducials: %s", self._visible_ids)

                # Pick master tag if it's in view
                target = next((f for f in fiducials if f.apriltag_properties.tag_id == self._master_tag_id), None)
                if target is not None:
                    winsound.PlaySound("SPOT following.wav", winsound.SND_FILENAME)
                    tf = get_a_tform_b(
                        target.transforms_snapshot,
                        VISION_FRAME_NAME,
                        target.apriltag_properties.frame_name_fiducial,
                        )
                    if tf is not None:
                        pos = tf.to_proto().position
                        LOGGER.info("Following master fiducial %d", self._master_tag_id)
                        self.go_to_tag(pos, locomotion_hint=spot_command_pb2.HINT_AUTO)
                    else:
                        LOGGER.info("Master tag detected but transform not available yet")

                elif self._follow_selected_only and self._selected_tag_id in self._visible_ids:
                    sel = self._selected_tag_id
                    target = next((f for f in fiducials
                                   if f.apriltag_properties.tag_id == sel), None)
                    if target is not None:
                        tf = get_a_tform_b(
                            target.transforms_snapshot,
                            VISION_FRAME_NAME,
                            target.apriltag_properties.frame_name_fiducial,
                        )
                        if tf is not None:
                            pos = tf.to_proto().position
                            if sel == 10:
                                LOGGER.info("Jogging to selected fiducial 10")
                                self.go_to_tag(pos, locomotion_hint=spot_command_pb2.HINT_JOG)
                            else:
                                LOGGER.info("Following selected fiducial %d", sel)
                                self.go_to_tag(pos, locomotion_hint=spot_command_pb2.HINT_AUTO)
                        else:
                            LOGGER.info("Selected tag %d transform not available yet", sel)
                    else:
                        LOGGER.info("Selected tag %d not in current fiducials", sel)

                else:
                    LOGGER.info("No target selected/visible (master not in view).")

            else:
                self._visible_ids = []
                LOGGER.info("No fiducials found")

            self._attempts += 1
            time.sleep(0.01)  # keep your original pacing

    def stop(self):
        self._movement_on = False
        self._stop.set()
        try:
            self._robot_command_client.robot_command(
                command=RobotCommandBuilder.stop_command())
        except Exception:
            pass

    def get_fiducial_objects(self):
        """Get all fiducials that Spot detects with its perception system."""
        # Get all fiducial objects (an object of a specific type).
        request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]
        fiducial_objects = self._world_object_client.list_world_objects(
            object_type=request_fiducials).world_objects
        if len(fiducial_objects) > 0:
            # Return the detected fiducials.
            return fiducial_objects
        # Return none if no fiducials are found.
        return None

    def power_on(self):
        """Power on the robot."""
        # If already powered, don't repeat.
        if self._robot.is_powered_on():
            self._powered_on = True
            print(f'Powered On {self._robot.is_powered_on()}')
            return
        self._robot.power_on()
        self._powered_on = True
        print(f'Powered On {self._robot.is_powered_on()}')

    def go_to_tag(self, fiducial_rt_world, locomotion_hint):
        """Use the position of the april tag in vision world frame and command the robot."""
        # Compute the go-to point (offset by .15m from the fiducial position) and the heading at
        # this point.
        self._current_tag_world_pose, self._angle_desired = self.offset_tag_pose(
            fiducial_rt_world, self._tag_offset)

        #Command the robot to go to the tag in kinematic odometry frame
        mobility_params = self.set_mobility_params()

        if mobility_params is not None and locomotion_hint is not None:
            mobility_params.locomotion_hint = locomotion_hint
            tag_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
                goal_x=self._current_tag_world_pose[0], goal_y=self._current_tag_world_pose[1],
                goal_heading=self._angle_desired, frame_name=VISION_FRAME_NAME, params=mobility_params,
                body_height=0.0)
        else:
            tag_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=self._current_tag_world_pose[0],
            goal_y=self._current_tag_world_pose[1],
            goal_heading=self._angle_desired,
            frame_name=VISION_FRAME_NAME,
            params=None,
            body_height=0.0,
            locomotion_hint=locomotion_hint)
            
        end_time = 5.0
        if self._movement_on and self._powered_on:
            #Issue the command to the robot
            self._robot_command_client.robot_command(lease=None, command=tag_cmd,
                                                     end_time_secs=time.time() + end_time)
            # #Feedback to check and wait until the robot is in the desired position or timeout
            start_time = time.time()
            current_time = time.time()
            while current_time - start_time < end_time:
                if self._stop.is_set() or not self._movement_on:
                    try:
                        self._robot_command_client.robot_command(
                            command=RobotCommandBuilder.stop_command())
                    except Exception:
                        pass
                    return
                if self.final_state():
                    return
                time.sleep(0.1)
                current_time = time.time()
        return

    def final_state(self):
        """Check if the current robot state is within range of the fiducial position."""
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
        """Offset the go-to location of the fiducial and compute the desired heading."""
        robot_rt_world = get_vision_tform_body(self.robot_state.kinematic_state.transforms_snapshot)
        robot_to_object_ewrt_world = np.array(
            [object_rt_world.x - robot_rt_world.x, object_rt_world.y - robot_rt_world.y, 0])
        norm = np.linalg.norm(robot_to_object_ewrt_world)
        if norm < 1e-6:
            # Already at the object — keep current heading.
            return np.array([robot_rt_world.x, robot_rt_world.y]), robot_rt_world.rot.to_yaw()
        robot_to_object_ewrt_world_norm = robot_to_object_ewrt_world / norm
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
                    obstacle_params=obstacles, vel_limit=speed_limit, body_control=body_control)
            else:
                mobility_params = spot_command_pb2.MobilityParams(
                    vel_limit=speed_limit, body_control=body_control)
        elif not self._avoid_obstacles:
            mobility_params = spot_command_pb2.MobilityParams(
                obstacle_params=obstacles, body_control=body_control)
        else:
            #When set to none, RobotCommandBuilder populates with good default values
            mobility_params = None
        return mobility_params
    
    def get_visible_tag_ids(self):
        """Return the list of currently visible fiducial IDs (ints)."""
        return list(self._visible_ids)

    @property
    def selected_tag_id(self):
        return self._selected_tag_id

    @property
    def follow_selected_only(self):
        return self._follow_selected_only

    def set_selected_tag(self, tag_id):
        """Directly set the selected tag (int or None)."""
        try:
            self._selected_tag_id = int(tag_id) if tag_id is not None else None
        except Exception:
            pass

    def cycle_selected(self, direction=+1):
        """Move selection to next/previous visible tag."""
        if not self._visible_ids:
            return
        # If current selection not visible, snap to the first visible
        if self._selected_tag_id not in self._visible_ids:
            self._selected_tag_id = self._visible_ids[0]
            return
        i = self._visible_ids.index(self._selected_tag_id)
        i = (i + direction) % len(self._visible_ids)
        self._selected_tag_id = self._visible_ids[i]

    def toggle_follow_selected(self):
        """Toggle follow-selected mode. Returns new boolean state."""
        self._follow_selected_only = not self._follow_selected_only
        # when user toggles back on, also resume motion
        if self._follow_selected_only:
            self._movement_on = True
        return self._follow_selected_only

    def cancel_motion_now(self):
        """Immediately stop whatever motion Spot is executing and pause following."""
        self._movement_on = False
        try:
            self._robot_command_client.robot_command(
                command=RobotCommandBuilder.stop_command()
            )
        except Exception:
            pass

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
        """Rotate the image so that it is always displayed upright."""
        if source_name == 'frontleft_fisheye_image':
            image = cv2.rotate(image, rotateCode=0)
        elif source_name == 'right_fisheye_image':
            image = cv2.rotate(image, rotateCode=1)
        elif source_name == 'frontright_fisheye_image':
            image = cv2.rotate(image, rotateCode=0)
        return image

class DisplayImagesAsync(object):
    """Display the images Spot sees from all five cameras."""

    def __init__(self, fiducial_follower, viewer_sources=None, scale=0.5):
        self._fiducial_follower = fiducial_follower
        self._image_client = fiducial_follower._image_client
        self._thread = None
        self._started = False
        self._stop = threading.Event()
        self._window_names = []

        all_sources = fiducial_follower.image_sources_list
        # Keep a small, useful default set to avoid heavy bandwidth; include hand_color if present.
        default_sources = [s for s in all_sources
                           if s in ['frontleft_fisheye_image',
                                    'frontright_fisheye_image',
                                    'left_fisheye_image',
                                    'right_fisheye_image',
                                    'back_fisheye_image']]
        self._sources = viewer_sources if viewer_sources else (default_sources or all_sources)
        self._scale = float(scale)

    def start(self):
        """Initialize the thread to display the images."""
        if self._started:
            return self
        self._stop.clear()
        self._started = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _fetch_bgr(self, source_name):
        try:
            # Request JPEG (color when available; mono remains mono).
            req = build_image_request(
                source_name,
                quality_percent=60,
                image_format=image_pb2.Image.FORMAT_JPEG
            )
            resp = self._image_client.get_image([req])[0]
        except Exception:
            return None

        # Decode bytes to BGR for display.
        buf = np.frombuffer(resp.shot.image.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return None

        # Rotate for correct orientation.
        bgr = self._fiducial_follower.rotate_image(bgr, source_name)

        # (Optional) keep a copy in follower for any other consumer
        self._fiducial_follower._image[source_name] = bgr
        return bgr

    def _run(self):
        idx = 0
        try:
            while not self._stop.is_set()  and len(self._sources) > 0:
                src = self._sources[idx % len(self._sources)]
                frame = self._fetch_bgr(src)
                if frame is not None:
                    if src not in self._window_names:
                        # Create named window so we can reliably destroy it.
                            cv2.namedWindow(src, cv2.WINDOW_NORMAL)
                            h0, w0 = frame.shape[:2]
                            cv2.moveWindow(src, int((idx % len(self._sources)) * w0 * self._scale), 0)
                            self._window_names.append(src)
                    h, w = frame.shape[:2]
                    resized = cv2.resize(frame, (int(w * self._scale), int(h * self._scale)),
                                         interpolation=cv2.INTER_NEAREST)
                    cv2.imshow(src, resized)
                    
                cv2.waitKey(1)
                idx += 1
                time.sleep(0.02)  # throttle a bit to avoid saturating link/CPU
        finally:
            # close just our windows from this thread
            for name in list(self._window_names):
                try:
                    cv2.destroyWindow(name)
                except Exception:
                    pass
            self._window_names.clear()
            # nudge HighGUI to process destroys
            cv2.waitKey(1)

    def stop(self):
        """Stop the thread and the image displays."""
        self._stop.set()
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.5)
            except Exception:
                pass
            self._thread = None
        self._started = False