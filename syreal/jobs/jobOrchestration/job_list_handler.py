from worker import Worker, WorkerManager, load_worker_manager
from typing import List, Dict, Union
from utils import Logger
logger, keep_fds = Logger(add_handler=False)


def clear_job_tickets(name=None, all=False):
    """Clears the job tickets directory. If name is specified, only the job ticket with that name is deleted. If all is True, all job tickets are deleted."""
    from constants import JOB_TICKETS
    for ticket in JOB_TICKETS.glob("*.json"):
        if all and not name:
            ticket.unlink()
        if name and (name == ticket.name.split('.')[0]):
            ticket.unlink()


def validate_status(status: List[str]) -> bool:
    """Checks if the status is valid. Returns True if the status is valid, False otherwise."""
    from constants import JOB_STATUS
    # check if the status is valid
    if not all([s in JOB_STATUS for s in status]):
        logger.error("Invalid status given to write_job_list_with_params.")
        return False
    return True


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
    for ticket in job_tickets[:num_tickets]:
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


def get_job_ticket(worker_name) -> Union[List[Dict], None]:
    """Returns the job ticket with the given name. If the job ticket does not exist, returns None."""
    from constants import JOB_TICKETS
    import json
    import os.path
    # read job ticket from job ticket directory
    ticket_path = JOB_TICKETS / f"{worker_name}.json"
    # if the job ticket does not exist, return None
    if not os.path.exists(ticket_path):
        logger.warning(
            f"Job ticket {worker_name} does not exist. Returning None.")
        return None
    # Read the job list
    with open(ticket_path, "r+") as file:
        # Load the JSON data from the file
        data = json.load(file)
    return data


def change_status_with_params(job_list, params, status, converged=None, iterations=None, exec_time=None):
    job_list["equations"][params["equation"]
                          ]["trials"][params["algorithm"]][str(params["trial"])]["status"] = status
    if converged is not None:
        job_list["equations"][params["equation"]
                              ]["trials"][params["algorithm"]][str(params["trial"])]["converged"] = converged
    if iterations is not None:
        job_list["equations"][params["equation"]
                              ]["trials"][params["algorithm"]][str(params["trial"])]["iterations"] = iterations
    if exec_time is not None:
        job_list["equations"][params["equation"]
                              ]["trials"][params["algorithm"]][str(params["trial"])]["exec_time"] = exec_time
    return job_list


def write_job_list(job_updates: List[Dict[str, str]], params_given=False, converged=None, iterations=None, exec_time=None):
    """
    Writes the job list to the job list file. job_updates is a list of dictionaries with the following keys: worker_name, name, status.

        Parameters:
            job_updates: A list of dictionaries with the following keys: worker_name, name, status if params_given is False. Otherwise, a list of dictionaries with the following keys: params, status.
        Returns:
            None
    """
    import json
    import fasteners
    from constants import JOB_LIST
    # from job ticket, get all parameters corresponding to the name to find the job in the job list
    if not params_given:
        # get all parameters from workers with given worker_names
        job_updates_zip = zip([job["worker_name"] for job in job_updates], [
            job["name"] for job in job_updates], [job["status"] for job in job_updates])
        job_updates = []
        WorkerManager = load_worker_manager()
        for worker_name, job_name, job_status in job_updates_zip:
            # get job info from WorkerManager
            # OLD METHOD: job_info = get_job_ticket(worker_name, job_name)
            worker = WorkerManager.get_worker(worker_name)
            job_info = worker.jobs[job_name]
            # if the job info is None, skip it
            if job_info:
                job_updates.append(dict(params=job_info, status=job_status))
        # validate status
        if not validate_status([job["status"] for job in job_updates]):
            return
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
            job_list = change_status_with_params(
                job_list, job["params"], job["status"], converged=converged, iterations=iterations, exec_time=exec_time)
            # find the job in the job list
        # Write the job list
        with open(JOB_LIST, "w+") as file:
            json.dump(job_list, file)


def validate_jobs(job_status_dict):
    from priority_queue import flatten_job_list, get_trials_from_ids
    # read the job list
    job_list = read_job_list()
    # flatten the job list
    flatten_jobs = list(flatten_job_list(
        job_list, skip_finished=False, skip_running=False))
    for job in job_status_dict:
        # get the trial from the job name
        trial_dict = get_trials_from_ids(job["name"], flatten_jobs)[0]
        # check if the job status is the same as the trial status
        if job["status"] != trial_dict[job["name"]]["status"]:
            yield False
        else:
            yield True


def find_status(job_list, params):
    status = job_list["equations"][params["equation"]
                                   ]["trials"][params["algorithm"]][params["trial"]]["status"]
    return status


def find_jobs_with_status(worker_name, status: Union[str, List[str]]):
    from priority_queue import flatten_job_list, get_trials_from_ids
    status = [status] if isinstance(status, str) else status
    # load the worker manager
    WorkerManager = load_worker_manager()
    # read the job list
    job_list = read_job_list()
    # get worker
    worker = WorkerManager.get_worker(worker_name)
    # get all jobs from the worker
    worker_jobs = worker.jobs
    for job in worker_jobs:
        # since the job is a dictionary, get the first value (the job parameters)
        params = list(job.values())[0]
        if find_status(job_list, params) in list(status):
            yield job


def update_jobs_status(worker_name: str, prev_status: str, new_status: str):
    """Takes a worker_name and updates only its jobs with status prev_status to new_status. Writes the updated job list to the job list file."""
    job_updates = list(find_jobs_with_status(worker_name, prev_status))
    # convert the job updates to a list of dictionaries with the following keys: params, status
    job_updates = [dict(params=list(job.values())[0], status=new_status) for job in job_updates]
    # write the job list
    write_job_list(job_updates, params_given=True)


def generate_data_dirs():
    """Generates the data directories for the worker output."""
    from constants import WORKER_OUTPUT_DIR
    # read the job list
    job_list = read_job_list()
    # for equation in job_list, for algorithm in equation, for trial in algorithm, make a directory
    for equation, value in job_list['equations'].items():
        for algorithm in value['trials'].keys():
            dir = WORKER_OUTPUT_DIR / equation / algorithm
            dir.mkdir(parents=True, exist_ok=True)

# TODO: update equation status (pending, running, finished)