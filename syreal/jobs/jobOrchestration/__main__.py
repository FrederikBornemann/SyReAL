# before importing anything else, create the directory structure. Some modules need it.
import sys
import argparse
from monitor import get_worker_number
from progress import get_progress
import os
from constants import PID_LOG_FILE, PID_DIR, PID_FILE, TEMP_DIR, JOB_TICKETS, PID_LOGS_DIR, WORKER_NUM, WORKER_LOGS_DIR, WORKER_OUTPUT_DIR, JOBS_DIR, JOB_LIST_BACKUP_DIR, PARALLEL_PROCS, SCHEDULER_INTERVAL, BATCH_DIR


def __setupDirectoryStructure():
    # if WORKER_OUTPUT_DIR does not exist or the directory has no subdirectories, raise an error
    if not os.path.exists(WORKER_OUTPUT_DIR) or len(os.listdir(WORKER_OUTPUT_DIR)) == 0:
        raise Exception(
            "The worker output directory does not exist or is empty. Please create and fill it. The default location is: " + str(WORKER_OUTPUT_DIR)) + ". The job orchestration will not work without it."
    requiredLocations = [JOBS_DIR, TEMP_DIR, PID_DIR, PID_LOGS_DIR,
                         JOB_TICKETS, WORKER_OUTPUT_DIR, WORKER_LOGS_DIR, JOB_LIST_BACKUP_DIR, BATCH_DIR]
    for loc in requiredLocations:
        if not os.path.exists(loc):
            os.makedirs(loc)
    # create log file
    if not os.path.exists(PID_LOG_FILE):
        with open(PID_LOG_FILE, "w") as f:
            f.write("")


# Make directories if they don't exist, because the modules need them
__setupDirectoryStructure()


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
    import daemonize
    from utils import Logger

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
    import time
    from monitor import get_SLURM_monitor, check_SLURM_monitor, print_jobs_as_table, get_worker_number
    import threading
    print("Monitoring job orchestration")
    # create a flag to stop the while loop
    stop_flag = threading.Event()
    # define your while loop function

    def while_loop():
        monitor = get_SLURM_monitor()
        while not stop_flag.is_set():
            monitor, changed, changed_rows, alerts = check_SLURM_monitor(
                monitor)
            print_jobs_as_table(monitor, alerts)
            print(
                f"Worker number: \033[1m\033[93m{get_worker_number()}/{WORKER_NUM}\033[0m")
            progress_finished, progress_running = get_progress(
                percent=True, round_to=5)
            num_finished, num_running, num_all = get_progress(percent=False)

            print(
                f"Progress: \033[1m\033[94m{progress_finished}\033[0m % finished ({num_finished}/{num_all}), \033[1m\033[92m{progress_running}\033[0m % running ({num_running})\033[0m")
            #print(f"ETA: \033[1m\033[92m{round(eta,1)}\033[0m minutes")
            print("\n\033[91mPress enter to exit.\033[0m")
            # check if the stop flag is set, if so break the loop. Do this in sleep because otherwise the loop will not break until the sleep is over
            sleep_time = SCHEDULER_INTERVAL
            for i in range(sleep_time):
                if stop_flag.is_set():
                    break
                time.sleep(1)

    # create a separate thread for the while loop
    while_thread = threading.Thread(target=while_loop)
    # start the while loop thread
    while_thread.start()
    # wait for the user to press enter
    input("Press enter to stop the loop...")
    # set the stop flag to stop the while loop
    stop_flag.set()
    # wait for the while loop thread to finish
    while_thread.join()


def stop():
    import signal
    # check if the daemon is running, if so stop it
    if not is_running():
        print("Job scheduler is not running.")
        return
    # Read the PID from the PID file
    with open(PID_FILE, "r") as f:
        pid = int(f.read().strip())
    # ask user if they want the daemon to finish the current job before stopping
    print("Stopping job scheduler...\n\t0: Stop the job scheduler immediately.\n\t1: Stop the job scheduler after the current job is finished.\n")
    userInput = input("Select an option: ")
    if userInput.lower() == "0":
        # Send a SIGTERM signal to the process
        print("Exiting immediately...")
        os.kill(pid, signal.SIGTERM)
    elif userInput.lower() == "1":
        # Send a SIGUSR1 signal to the process
        print("Stopping job scheduler, exiting after the current job is finished...")
        os.kill(pid, signal.SIGUSR1)


########## MAIN ##########

# Define actions, help, and colors
actions = [start, stop, monitor]
actionHelp = ["Start the job scheduler",
              "Stop the job scheduler (stop or kill jobs)", "Monitor the jobs"]
colors = ["\033[92m", "\033[91m", "\033[93m"]
# Print status table
# TODO: add progress percentage
print("\nSTATUS\t\tWORKERS\t\tPROGRESS [finished] (running)")
progress_finished, progress_running = get_progress(percent=True, round_to=1)
if is_running():
    print(
        f"\033[92mRunning\033[0m\t\t({get_worker_number()}/{WORKER_NUM})\t\t[{progress_finished} %] ({progress_running} %)\n")
else:
    print(
        f"\033[91mStopped\033[0m\t\t({get_worker_number()}/{WORKER_NUM})\t\t[{progress_finished} %] ({progress_running} %)\n")
parser = argparse.ArgumentParser()
parser.add_argument("-action", type=str)
args = parser.parse_known_args()[0]
if "action" in args and args.action is not None:
    actionId = args.action.lower()
else:
    # print in bold text
    print("\033[1m" + "Choose an action:" + "\033[0m")
    for i, act in enumerate(actions):
        print(str(i) + ": " + colors[i] + act.__name__ +
              "\033[0m" + " - " + actionHelp[i])
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
