from job_monitor import get_status_of_jobs
import json
import pickle
from constants import TEMP_DIR, JOB_DICT, WORKER_OUTPUT_DIR
from tqdm import tqdm
from monitor import get_worker_names


def get_job_status(save_to_file=True, new_status=False, state="finished"):
    """Get the status of all the jobs in the Worker output directory. If they are finished, read the parameters.json file. Return a dictionary with keys for every equation."""
    if (not JOB_DICT.exists()) or new_status:
        status = get_status_of_jobs()
        with open(JOB_DICT, "wb") as f:
            pickle.dump(status, f)
    else:
        with open(JOB_DICT, "rb") as f:
            status = pickle.load(f)

    # delete all the jobs that are not finished
    jobs_with_state = []
    failed_job_parameters = []
    total_iterations = len(status)
    # Create a progress bar using tqdm
    progress_bar = tqdm(total=total_iterations, unit='iteration')
    for job in status:
        # status[job] is a tuple with the status and the parameters

        if status[job][0] != state:
            continue

        # read the parameters.json file.
        # Update the progress bar
        progress_bar.update(1)
        try:
            with open(WORKER_OUTPUT_DIR / job[0] / job[1] / job[2] / "parameters.json", "r") as f:
                parameters = json.load(f)
            # add the parameters to the jobs list
            jobs_with_state.append((job, parameters))
            continue
        except:
            # add the job to the failed jobs list
            if state == "finished":
                failed_job_parameters.append(job)
                continue
            else:
                # add the job to the jobs list
                jobs_with_state.append((job, None))
                continue
        

    # Close the progress bar
    progress_bar.close()

    print(f"Found {len(jobs_with_state)} {state} jobs")
    print(f"Found {len(failed_job_parameters)} failed jobs")
    print(failed_job_parameters)
    # save the status to a file
    if save_to_file:
        filename = TEMP_DIR / f"{state}_jobs.json"
        with open(filename, "w") as f:
            json.dump(jobs_with_state, f)


def eq_algo_summary(save_to_file=True):
    """Summarizes the number of finished jobs for each equation and algorithm. Returns a dictionary with the number of finished jobs and average last_n for each equation and algorithm."""
    # read the finished jobs file
    with open(TEMP_DIR / "finished_jobs.json", "r") as f:
        finished_jobs = json.load(f)
    # create a dictionary with the number of finished jobs and average last_n for each equation and algorithm
    eq_algo_summary = {}
    for job in finished_jobs:
        # job[0] is the job name, job[1] is the parameters
        equation = job[0][0]
        algorithm = job[0][1]
        last_n = job[1]["last_n"]
        #converged = job[1]["converged"]
        # if the equation is not in the dictionary, add it
        if equation not in eq_algo_summary:
            eq_algo_summary[equation] = {}
        # if the algorithm is not in the dictionary, add it
        if algorithm not in eq_algo_summary[equation]:
            eq_algo_summary[equation][algorithm] = {
                "finished": 0, "last_n": []}
        # add the last_n to the list of last_n
        eq_algo_summary[equation][algorithm]["last_n"].append(last_n)
        # add 1 to the number of finished jobs
        eq_algo_summary[equation][algorithm]["finished"] += 1
    # calculate the average last_n for each equation and algorithm
    for equation in eq_algo_summary:
        for algorithm in eq_algo_summary[equation]:
            eq_algo_summary[equation][algorithm]["average_last_n"] = sum(
                eq_algo_summary[equation][algorithm]["last_n"]) / eq_algo_summary[equation][algorithm]["finished"]
    if save_to_file:
        with open(TEMP_DIR / "eq_algo_summary.json", "w") as f:
            json.dump(eq_algo_summary, f)
    return eq_algo_summary


if __name__ == "__main__":
    # get_job_status()
    # eq_algo_summary()
    #get_job_status(state="stopped", new_status=True)
    get_job_status(state="finished")
    #get_job_status(state="running")
    eq_algo_summary()
