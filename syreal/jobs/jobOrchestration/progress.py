from job_list_handler import read_job_list
from priority_queue import flatten_job_list
from constants import PARALLEL_PROCS, WORKER_NUM
from numpy import median, mean


def get_progress(percent=False, ETA=False):
    job_list = read_job_list()
    flatten_jobs = list(flatten_job_list(job_list, skip_finished=False, skip_running=False))
    # get the number finished jobs, running jobs, and all jobs
    all_jobs = len(flatten_jobs)
    finished_jobs = [list(job.values())[0]["exec_time"] for job in flatten_jobs if list(job.values())[0]["status"] == "finished"]
    # calculate ETA
    if len(finished_jobs) == 0:
        average_time = 0
    else:
        # ignore None values in time 
        average_time = mean([int(time.split(":")[0]) * 3600 + int(time.split(":")[1]) * 60 + int(time.split(":")[2]) for time in finished_jobs if time is not None])
    finished_jobs = len(finished_jobs)
    running_jobs = len([job for job in flatten_jobs if list(job.values())[0]["status"] == "running"])
    eta = average_time*(all_jobs-finished_jobs)/(60*WORKER_NUM*PARALLEL_PROCS*0.75)
    # TODO: eta is highly inaccurate. Fix it.
    if percent:
        if ETA:
            return finished_jobs / all_jobs * 100, running_jobs / all_jobs * 100, eta
        return finished_jobs / all_jobs * 100, running_jobs / all_jobs * 100
    if ETA:
        return finished_jobs, running_jobs, all_jobs, eta
    return finished_jobs, running_jobs, all_jobs
