# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

""" Trying to create a basic program """

import argparse
import curses
import logging
import os
import signal
import sys
import time

import bosdyn.client.util
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.robot_state import RobotStateClient

class BasicProgram():
    """ Basic program class """
    def __init__(self, client, timeout_sec, name=None):
        self.robot = robot

