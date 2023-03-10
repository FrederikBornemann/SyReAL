

from pathlib import Path
import json
import hashlib
import time


# make a time function to use as a decorator
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time taken by {func.__name__}: {end - start} seconds")
        return result
    return wrapper


def generate_trial_id(param1, param2, param3):
    # Generate a unique ID for a trial
    # Concatenate the parameters into a single string
    param_string = str(param1) + str(param2) + str(param3)
    # Create a SHA-256 hash object
    hash_object = hashlib.sha256()
    # Encode the parameter string as bytes and update the hash object
    hash_object.update(param_string.encode())
    # Get the hexadecimal representation of the hash
    hex_digest = hash_object.hexdigest()
    # Return the first 8 characters of the hash as the ticket ID
    return hex_digest[:8]


def flatten_job_list(job_list, skip_finished=True, skip_running=True):
    # Flatten the job list
    for eq in job_list["equations"]:
        eq_dict = job_list["equations"][eq]
        # skip finished equations
        if skip_finished and eq_dict["status"] == "finished":
            continue
        for algo in eq_dict["trials"]:
            algo_dict = eq_dict["trials"]
            for trial in algo_dict[algo]:
                trial_dict = algo_dict[algo][trial]
                if skip_finished and trial_dict["status"] == "finished":
                    continue
                if skip_running and trial_dict["status"] == "running":
                    continue
                yield {
                    generate_trial_id(trial, eq, algo): {
                        "trial": trial,
                        "equation": eq,
                        "algorithm": algo,
                        "status": trial_dict["status"],
                        "equation_status": eq_dict["status"],
                        # "progress": trial_dict["progress"],
                    }
                }


def priority(job):
    # Priority is determined by the status of the equation and the trial
    if job["equation_status"] == "running":
        if job["status"] == "stopped":
            return 0
        elif job["status"] == "pending":
            return 1
    elif job["equation_status"] == "pending":
        if job["status"] == "stopped":
            return 2
        elif job["status"] == "pending":
            return 3
    return 999


def get_trials_from_ids(trial_ids, flatten_job_list):
    # get the trial dicts from the trial ids
    trial_dicts = []
    for trial in flatten_job_list:
        id = list(trial.keys())[0]
        if id in trial_ids:
            trial_dicts.append(trial)
    return trial_dicts
    # the time complexity of this is O(n^2) but it's not a big deal


def make_priority_queue(job_list, number_of_jobs=5):
    # Flatten the job list
    flatten_jobs = list(flatten_job_list(job_list))
    # Create a priority queue
    priority_queue = []
    for ticket in flatten_jobs:
        id = list(ticket.keys())[0]
        job = ticket[id]
        # Append the job to the priority queue
        priority_queue.append(dict(id=id, priority=priority(job)))
    # Sort the priority queue
    priority_queue.sort(key=lambda x: x["priority"])
    # Return the first n jobs from their ids
    trials = get_trials_from_ids([priority_queue[i]["id"]
                                 for i in range(number_of_jobs)], flatten_jobs)
    return trials


DIR = Path(__file__).parents[0]
with open(DIR/"jobs.json", "r") as f:
    job_list = json.load(f)

trials = make_priority_queue(job_list, number_of_jobs=2)
print(trials)
