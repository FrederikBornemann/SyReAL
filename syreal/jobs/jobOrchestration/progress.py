from job_list_handler import read_job_list
from priority_queue import flatten_job_list


def get_progress(percent=False):
    job_list = read_job_list()
    flatten_jobs = list(flatten_job_list(job_list, skip_finished=False, skip_running=False))
    # get the number finished jobs, running jobs, and all jobs
    all_jobs = len(flatten_jobs)
    finished_jobs = len([job for job in flatten_jobs if list(job.values())[0]["status"] == "finished"])
    running_jobs = len([job for job in flatten_jobs if list(job.values())[0]["status"] == "running"])
    if percent:
        return finished_jobs / all_jobs * 100, running_jobs / all_jobs * 100
    return finished_jobs, running_jobs, all_jobs

def ETA():
    # TODO: implement this
    finished, running = get_progress()
    # get the time
    # calculate the ETA
    pass

# Write a function that returns the eta for all jobs