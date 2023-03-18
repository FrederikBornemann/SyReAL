import argparse
import os
import sys
from pathlib import Path

from constants import PID_DIR, PID_FILE, TEMP_DIR, JOB_TICKETS, PID_LOG_FILE, PID_LOGS_DIR

import daemonize
import signal
import logging

from utils import Logger


def __setupDirectoryStructure():
    requiredLocations = [TEMP_DIR, PID_DIR, PID_LOGS_DIR, JOB_TICKETS]
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
        # Set up logging
        logger, keep_fds = Logger()
        daemon = daemonize.Daemonize(
            app="job_scheduler", pid=PID_FILE, action=start_job_orchestrator, logger=logger, verbose=True, keep_fds=keep_fds)
        daemon.start()
    # check if the daemon is already running, if not start it
    if is_running():
        print("Daemon is already running.")
    else:
        start_daemon()


def monitor():
    print("Monitoring job orchestration")
    # TODO: print the monitor, wait for user input, and then exit1


def stop():
    # check if the daemon is running, if so stop it
    if not is_running():
        print("Job scheduler is not running.")
        return
    
    # Read the PID from the PID file
    with open(PID_FILE, "r") as f:
        pid = int(f.read().strip())
    # ask user if they want the daemon to finish the current job before stopping
    userInput = input("Do you want the job scheduler to finish the current jobs before stopping? (y/n): ")
    if userInput.lower() == "y" or userInput.lower() == "yes" or userInput.lower() == "":
        # Send a SIGTERM signal to the process
        os.kill(pid, signal.SIGUSR1)
    elif userInput.lower() == "n" or userInput.lower() == "no":
        # ask for confirmation
        userInput = input("Are you sure you want to kill all running jobs? (y/n): ")
        if userInput.lower() == "y" or userInput.lower() == "yes":
            # Send a SIGKILL signal to the process
            os.kill(pid, signal.SIGTERM)
        else:
            print("Daemon was not stopped.")

actions = [start, stop, monitor]
actionHelp = ["Start the job scheduler", "Stop the job scheduler. (terminate or kill)", "Monitor the job scheduler"]
colors = ["\033[92m", "\033[91m", "\033[93m"]
# print status of daemon in green if running, red if not
if is_running():
    print("\nJob-Scheduler status: \033[92mrunning\033[0m\n")
else:
    print("\nJob-Scheduler status: \033[91mnot running\033[0m\n")
parser = argparse.ArgumentParser()
parser.add_argument("-action", type=str)
args = parser.parse_known_args()[0]
if "action" in args and args.action is not None:
    actionId = args.action.lower()
else:
    # print in bold text
    print("\033[1m" + "Choose an action:" + "\033[0m")
    for i, act in enumerate(actions):
        print(str(i) + ": " + colors[i] + act.__name__ + "\033[0m" + " - " + actionHelp[i])
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

