# TODO
# [] read job list
# [] make job tickets
# [] spawn jobs
# [] update job list


# This is the main job orchestrator. It is responsible for spawning jobs and updating the job list.
# It is also responsible for spawning the job monitor, which is responsible for monitoring the jobs

import os
import sys
import time
import json
import fcntl
from pathlib import Path
import fasteners

from .priority_queue import make_priority_queue, get_trials_from_ids


PKG_DIR = Path(__file__).parents[3]
JOBS_DIR = os.environ.get('JOB_ORCHESTRATION_WORKSPACE')
JOB_LIST = JOBS_DIR / "job_list.json"


def read_job_list():
    # Read the job list
    with open(JOB_LIST, "r+") as file:
        # Load the JSON data from the file
        data = json.load(file)
    return data


def write_job_ticket(job_list, num_jobs):
    # Create a job ticket
    job_ticket = {
        "job_id": len(job_list),
        "job_name": f"job_{len(job_list)}",
        "mincpus": 1,
        "scripts": f"python {PKG_DIR}/syreal/jobs/distributor.py --num_jobs {num_jobs}"
    }
    
    # Append the job ticket to the job list
    job_list.append(job_ticket)
    
    # Write the job list
    with open(JOB_LIST, "w") as file:
        json.dump(job_list, file, indent=4)
    
    return job_list

if __name__ == "__main__":
    # Create a file lock
    lock = fasteners.InterProcessLock(JOB_LIST + ".lock")
    # Acquire the lock
    with lock:
        # Read the job list
        job_list = read_job_list()
    
    # make job tickets
    queue = make_priority_queue(job_list)
    
    
    
    



