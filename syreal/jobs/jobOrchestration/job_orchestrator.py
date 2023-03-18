# TODO
# [x] read job list
# [x] make job tickets
# [] spawn jobs
# [] update job list


# This is the main job orchestrator. It is responsible for spawning jobs and updating the job list.
# It is also responsible for spawning the job monitor, which is responsible for monitoring the jobs

import os
import subprocess
import json
from pathlib import Path
import fasteners
import pandas as pd

from priority_queue import make_priority_queue, get_trials_from_ids
from id_generation import generate_id
from monitor import check_SLURM_monitor, get_SLURM_monitor, print_jobs_as_table


from constants import JOB_LIST, JOB_TICKETS, SLURM_USER, SLURM_PARTITION


def read_job_list():
    # Read the job list
    with open(JOB_LIST, "r+") as file:
        # Load the JSON data from the file
        data = json.load(file)
    return data


def clear_job_tickets():
    # remove all job tickets
    for ticket in JOB_TICKETS.glob("*.json"):
        ticket.unlink()

def write_job_tickets(queue, num_jobs):
    # save all portions of the queue with length num_jobs as a job ticket
    # split the queue into parts of length num_jobs
    job_tickets = [queue[i:i + num_jobs] for i in range(0, len(queue), num_jobs)]
    # save the job tickets to the job_tickets directory. If the directory does not exist, create it
    if not JOB_TICKETS.exists():
        JOB_TICKETS.mkdir()
    # write the job tickets to the job_tickets directory
    for ticket in job_tickets:
        all_ids = [list(job.keys())[0] for job in ticket]
        id = generate_id(*all_ids, n=8)
        with open(JOB_TICKETS / f"job_ticket_{id}.json", "w+") as file:
            json.dump(ticket, file)
    
def start_job_orchestrator():
    import signal
    import sys

    def sigterm_handler(signum, frame):
        print("Received SIGTERM signal. Exiting...")
        sys.exit(0)

    # Set the signal handler for SIGTERM
    signal.signal(signal.SIGTERM, sigterm_handler)

    # Start the job orchestrator
    pass

if __name__ == "__main__":
    # Create a file lock
    lock = fasteners.InterProcessLock(str(JOB_LIST) + ".lock")
    # Acquire the lock
    with lock:
        # Read the job list
        job_list = read_job_list()
    
    # make job tickets
    queue = make_priority_queue(job_list)
    # clear job tickets
    clear_job_tickets()
    # write job tickets
    write_job_tickets(queue, num_jobs=2)
    # monitor
    # get the SLURM monitor
    monitor = get_SLURM_monitor()
    while True:
        # check the SLURM monitor
        monitor, changed, changed_rows, alerts = check_SLURM_monitor(monitor)
        # print the jobs as a table
        print_jobs_as_table(monitor, alerts)
    
    
    
    



