import argparse
import os
import sys
from pathlib import Path

from constants import PID_DIR, PID_FILE, TEMP_DIR, JOB_TICKETS

import daemonize
import signal


def __setupDirectoryStructure():
    requiredLocations = [TEMP_DIR, PID_DIR, JOB_TICKETS]
    for loc in requiredLocations:
        if not os.path.exists(loc):
            os.makedirs(loc)

# Check if the daemon is running
def is_running():
        if os.path.exists(PID_FILE):
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, 0)  # Check if the process is running
            except OSError:
                return False
            else:
                return True
        else:
            return False

def start():
    def start_daemon():
        from job_orchestrator import start_job_orchestrator
        print("Starting job orchestration")
        daemon = daemonize.Daemonize(
            app="mydaemon", pid=PID_FILE, function=start_job_orchestrator)
        daemon.start()
    # check if the daemon is already running, if not start it
    if is_running():
        print("Daemon is already running.")
    else:
        start_daemon()


def monitor():
    print("Monitoring job orchestration")


def stop():
    # check if the daemon is running, if so stop it
    if not is_running():
        print("Daemon is not running.")
        return
    print("Stopping job orchestration")
    # Read the PID from the PID file
    with open(PID_FILE, "r") as f:
        pid = int(f.read().strip())
    # Send a SIGTERM signal to the process
    os.kill(pid, signal.SIGTERM)


actions = [start, monitor, stop]

parser = argparse.ArgumentParser()
parser.add_argument("-action", type=str)
args = parser.parse_known_args()[0]
if "action" in args and args.action is not None:
    actionId = args.action.lower()
else:
    print("Which action would you like to perform?")
    for i, act in enumerate(actions):
        print("    {}: {}".format(i, act.__name__))
    userInput = input()
    if str.isdigit(userInput):
        actionId = actions[int(userInput)].__name__.lower()
    else:
        actionId = userInput.lower()

if actionId in ["", "na"]:
    __setupDirectoryStructure()
    sys.exit(0)

for action in actions:
    if action.__name__.lower() == actionId:
        __setupDirectoryStructure()
        action()
        break
else:
    raise Exception("-action must be in following list " +
                    str([x.__name__ for x in actions]))
