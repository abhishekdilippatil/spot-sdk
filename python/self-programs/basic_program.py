# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

""" Trying to create a basic program """

import argparse
import sys
import time

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.image import ImageClient


class EStop():
    """Manages a simple E-Stop configuration and keep-alive for the program."""

    def __init__(self, client, timeout_sec, name=None):
        self.logger = bosdyn.client.util.get_logger()
        if name is None:
            name = "EStop"

        # Force server to set up a single endpoint system
        self.estop_endpoint = EstopEndpoint(client, name, timeout_sec)
        self.estop_endpoint.force_simple_setup() # This makes this client the sole E-Stop source.
        self.logger.info("E-Stop endpoint forced simple setup.")

        # Begin periodic check-in between keep-alive and robot
        rpc_interval_secs = timeout_sec / 3.0
        self.estop_keep_alive = EstopKeepAlive(self.estop_endpoint, rpc_interval_seconds=rpc_interval_secs)
        self.logger.info("E-Stop keep-alive started.")

        # Release the estop
        self.estop_keep_alive.allow()
        self.logger.info("E-Stop allowed.")

    def settle_then_cut(self):
        if self.estop_keep_alive:
            self.logger.info("Commanding E-Stop: Settle then Cut.")
            self.estop_keep_alive.settle_then_cut()

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args()
    logger = bosdyn.client.util.get_logger() # Get logger for main function

    def basic_program(config):
        """This is a basic program"""

        bosdyn.client.util.setup_logging(config.verbose)

        # SDK object is created and is a primamry interface to the API.
        # This will initialize typical default parameters
        # 
        sdk = bosdyn.client.create_standard_sdk('BasicProgram') 

        # Create robot object
        # Network address needs to be provided.
        robot = sdk.create_robot(config.hostname)

        # Client needs authentication before use.
        bosdyn.client.util.authenticate(robot)

        # Time sync is required to issue commands to the robot.
        robot.time_sync.wait_for_sync()

        # Setup E-Stop management for this program
        client = robot.ensure_client(EstopClient.default_service_name)
       
        # Default E-Stop timeout: 60 seconds. Robot will E-Stop if it doesn't get a check-in.
        estop_manager = EStop(client, timeout_sec=60.0)

        try:
            # RobotStateClient allows to get robot state information
            robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

            # LeaseClient is used to acquire lease
            lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
            with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
                
                #Powering on the robot
                robot.logger.info('The dog is waking up!')
                #timeout set for response from robot
                robot.power_on(timeout_sec=20)
                #Check if robot is powered on
                assert robot.is_powered_on(), 'Dog did not wake up'
                robot.logger.info('The dog is awake')

                robot.logger.info('Getting ready for walkies...')

                command_client = robot.ensure_client(RobotCommandClient.default_service_name)
                #helper function provided by Boston Dynamics to command robot to stand up
                blocking_stand(command_client, timeout_sec=10)
                robot.logger.info('Doggo is standing')
                #Execution of script for 5 seconds, before moving onto next command.
                time.sleep(3)     
                
                image_client = robot.ensure_client(ImageClient.default_service_name)
                sources = image_client.list_image_sources() # pylint: disable=unused-variable
                image_response = image_client.get_image_from_sources(['left_fisheye_image'])
                display_image(image_response[0].shot.image)

                robot.logger.info("Powering off robot...")
                robot.power_off(cut_immediately=False, timeout_sec=20)
                assert not robot.is_powered_on(), "Robot power off failed."
                robot.logger.info("Robot safely powered off.")
            # Lease is automatically returned here by LeaseKeepAlive.__exit__
        finally:
            # Ensure E-Stop is cleanly handled: command a cut and stop the keepalive thread.
            estop_manager.settle_then_cut()
            estop_manager.estop_keep_alive.shutdown() # Explicitly stop the E-Stop keepalive thread
            robot.logger.info("E-Stop shut down.")
            
    def display_image(image, display_time = 3.0):
        """Capturing and displaying image"""
        try:
            import io
            from PIL import Image # type: ignore
        except ImportError:
            logger = bosdyn.client.util.get_logger()
            logger.warning('Missing dependencies. can\'t display image')
            return
        try:
            image = Image.open(io.BytesIO(image.data))
            image.show()
            time.sleep(display_time)
        except Exception as exc:
            logger = bosdyn.client.util.get_logger()
            logger.warning('Exception thrown displaying image. %r', exc)


    try:
        basic_program(options)
        return True
    
    except Exception as exc:  # pylint: disable=broad-except
        logger.error('Basic program threw an exception: %r', exc)
        return False
    
if __name__ == '__main__':
    if not main():
        sys.exit(1)