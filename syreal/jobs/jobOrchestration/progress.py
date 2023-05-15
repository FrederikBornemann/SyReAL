from constants import PARALLEL_PROCS, WORKER_NUM, PROGRESS_FILE
from numpy import median, mean
import json


# def get_progress(percent=False, ETA=False) -> tuple:
#     """Returns the progress of the job list. If percent is True, returns the progress as a percentage. If ETA is True, returns the ETA in minutes."""
#     job_list = read_job_list()
#     flatten_jobs = list(flatten_job_list(job_list, skip_finished=False, skip_running=False))
#     # get the number finished jobs, running jobs, and all jobs
#     all_jobs = len(flatten_jobs)
#     finished_jobs = [list(job.values())[0]["exec_time"] for job in flatten_jobs if list(job.values())[0]["status"] == "finished"]
#     # calculate ETA
#     if len(finished_jobs) == 0:
#         average_time = 0
#     else:
#         # ignore None values in time 
#         average_time = mean([int(time.split(":")[0]) * 3600 + int(time.split(":")[1]) * 60 + int(time.split(":")[2]) for time in finished_jobs if time is not None])
#     finished_jobs = len(finished_jobs)
#     running_jobs = len([job for job in flatten_jobs if list(job.values())[0]["status"] == "running"])
#     eta = average_time*(all_jobs-finished_jobs)/(60*WORKER_NUM*PARALLEL_PROCS*0.75)
#     # TODO: eta is highly inaccurate. Fix it.
#     if percent:
#         if ETA:
#             return finished_jobs / all_jobs * 100, running_jobs / all_jobs * 100, eta
#         return finished_jobs / all_jobs * 100, running_jobs / all_jobs * 100
#     if ETA:
#         return finished_jobs, running_jobs, all_jobs, eta
#     return finished_jobs, running_jobs, all_jobs

def get_progress(percent=True, round_to=2) -> tuple:
    """Returns the progress of the jobs. If percent is True, returns the progress as a percentage. Otherwise, returns the number of finished jobs, running jobs, and all jobs. Round to specifies the number of decimal places to round to."""
    if not PROGRESS_FILE.exists():
        return (0, 0) if percent else (0, 0, 0) # if the progress file doesn't exist, return 0
    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)
    num_finished = progress["finished"]
    num_running = progress["running"]
    num_all = progress["pending"] + progress["stopped"] + num_finished + num_running
    if percent:
        if num_all == 0:
            return 0, 0 # otherwise we get a division by zero error
        return round(num_finished / num_all * 100, round_to), round(num_running / num_all * 100, round_to)
    return num_finished, num_running, num_all


if __name__ == "__main__":
    print(get_progress(percent=False, round_to=1))