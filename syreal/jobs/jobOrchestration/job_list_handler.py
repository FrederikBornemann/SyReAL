from utils import Logger
logger, keep_fds = Logger(add_handler=False)
from typing import List, Dict


def clear_job_tickets(name=None, all=False):
    """Clears the job tickets directory. If name is specified, only the job ticket with that name is deleted. If all is True, all job tickets are deleted."""
    from constants import JOB_TICKETS
    for ticket in JOB_TICKETS.glob("*.json"):
        if all and not name:
            ticket.unlink()
        if name and (name == ticket.name.split('.')[0]):
            ticket.unlink()


def write_job_tickets(queue, num_jobs, num_tickets):
    from constants import JOB_TICKETS
    from utils import generate_id
    import json
    # save all portions of the queue with length num_jobs as a job ticket
    # split the queue into parts of length num_jobs
    job_tickets = [queue[i:i + num_jobs]
                   for i in range(0, len(queue), num_jobs)]
    # save the job tickets to the job_tickets directory. If the directory does not exist, create it
    if not JOB_TICKETS.exists():
        JOB_TICKETS.mkdir()
    # write the job tickets to the job_tickets directory
    for ticket in job_tickets[:num_tickets-1]:
        all_ids = [list(job.keys())[0] for job in ticket]
        id = generate_id(*all_ids, n=8)
        with open(JOB_TICKETS / f"{id}.json", "w+") as file:
            json.dump(ticket, file)
        yield dict(id=id, jobs=ticket)


def read_job_list():
    import json
    import fasteners
    from constants import JOB_LIST
    # Create a file lock
    lock = fasteners.InterProcessLock(str(JOB_LIST) + ".lock")
    # Acquire the lock
    with lock:
        # Read the job list
        with open(JOB_LIST, "r+") as file:
            # Load the JSON data from the file
            data = json.load(file)
        return data
    

def get_job_ticket(worker_name, job_name):
    from constants import JOB_TICKETS
    import json
    import os.path
    # read job ticket from job ticket directory
    ticket_path = JOB_TICKETS / f"{worker_name}.json"
    # if the job ticket does not exist, return None   
    if not os.path.exists(ticket_path):
        logger.debug(f"Job ticket {worker_name} does not exist. But it was requested by {job_name}. It was probably deleted.")
        return None
    # Read the job list
    with open(ticket_path, "r+") as file:
        # Load the JSON data from the file
        data = json.load(file)
    for job in data:
        if job_name in job:
            return job[job_name]
    # if the job does not exist in the job ticket, return None
    logger.debug(f"Job {job_name} does not exist in job ticket {worker_name}.")
    return None


def change_status_with_params(job_list, params, status):
    job_list["equations"][params["equation"]]["trials"][params["algorithm"]][params["trial"]]["status"] = status
    return job_list


def write_job_list(job_updates: List[Dict[str, str]]):
    import json
    import fasteners
    from constants import JOB_LIST
    # from job ticket, get all parameters corresponding to the name to find the job in the job list
    job_updates_zip = zip([job["worker_name"] for job in job_updates], [job["name"] for job in job_updates], [job["status"] for job in job_updates])
    job_updates = []
    for worker_name, job_name, job_status in job_updates_zip:
        # get job info from job ticket
        job_info = get_job_ticket(worker_name, job_name)
        # if the job info is None, skip it
        if job_info:
            job_updates.append(dict(params=job_info, status=job_status))
    # Create a file lock
    lock = fasteners.InterProcessLock(str(JOB_LIST) + ".lock")
    # Acquire the lock
    with lock:
        # Read the job list
        with open(JOB_LIST, "r+") as file:
            # Load the JSON data from the file
            job_list = json.load(file)
        # Update the job list
        for job in job_updates:
            job_list = change_status_with_params(job_list, job["params"], job["status"])
            # find the job in the job list
        # Write the job list
        with open(JOB_LIST, "w+") as file:
            json.dump(job_list, file)


def validate_jobs(job_status_dict):
    from priority_queue import flatten_job_list, get_trials_from_ids
    # read the job list
    job_list = read_job_list()
    # flatten the job list
    flatten_jobs = list(flatten_job_list(job_list, skip_finished=False, skip_running=False))
    for job in job_status_dict:
        # get the job status
        job_status = job["status"]
        # get the job ticket
        job_ticket = job["name"]
        # get the trial dicts from the trial ids
        trial_dict = get_trials_from_ids(job_ticket, flatten_jobs)[0]
        # check if the job status is the same as the trial status
        if job_status != trial_dict[job_ticket]["status"]:
            yield False
        else:
            yield True
            