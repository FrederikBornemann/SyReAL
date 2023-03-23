from utils import generate_id

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
                    generate_id(trial, eq, algo, n=8): {
                        "trial": trial,
                        "equation": eq,
                        "algorithm": algo,
                        "status": trial_dict["status"],
                        "equation_status": eq_dict["status"],
                        "equation_params": eq_dict["parameters"],
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


def make_priority_queue(job_list, number_of_jobs):
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

