import os
import subprocess

"""Script to run program on a robot using the Spot SDK."""

# Configuration
program_directory = os.path.dirname(os.path.abspath(__file__))
program_name = "fiducial_wasd.py"
robot_ip = "192.168.80.3"

# Change to the directory where the program is located
os.chdir(program_directory)

# Run the program
subprocess.run(['cmd', '/c', 'start', 'cmd', '/k', 'python3', program_name, robot_ip])