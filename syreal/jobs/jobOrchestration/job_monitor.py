import pickle
import json

from constants import WORKER_OUTPUT_DIR, JOB_DICT
from monitor import get_worker_names
from utils import timeit_wrapper

@timeit_wrapper
def get_status_of_jobs() -> dict:
    """
    Returns a dictionary of the status of all jobs. Goes through the files of the job files and checks the status of each job.
    The status of each job is either "finished", "running", "stopped" or "pending".
    """

    # File hierarchy:
    # WORKER_OUTPUT_DIR
    #   - Num equation
    #       - algorithm
    #           - trial
    #               - parameters.json
    #               - Done.txt
    #               - Running.txt
    #               - output (csv etc.)

    def get_status_of_job(equation, algorithm, trial, running_worker_names):
        """
        Returns the status of a job. The status of each job is either "finished", "running", "stopped" or "pending".
        - If "finished" then the job directory contains the "Done.txt" file.
        - If "running" then the job directory contains the "Running.txt" file and the "Done.txt" file does not exist. Also the worker_name in the "Running.txt" file is the same as a running worker.
        - If "stopped" then the job directory contains the "Running.txt" file and the "Done.txt" file does not exist. The worker_name in the "Running.txt" file is not the same as any running worker.
        - If "pending" then the job directory does not contain the "Running.txt" file and the "Done.txt" file does not exist.
        """
        run_folder = WORKER_OUTPUT_DIR / equation / algorithm / trial
        # Check if the job is finished
        if (run_folder / "Done.txt").exists():
            return "finished"
        # Check if the job is running
        elif (run_folder / "Running.txt").exists():
            # Get the worker_id of the running job
            with open(run_folder / "Running.txt", "r") as f:
                worker_name = f.read().strip()
            # Get the worker_id of the running jobs
            running_worker_names = get_worker_names()
            # Check if the worker_name of the running job is in the running_worker_names
            if worker_name in running_worker_names:
                return "running"
            else:
                return "stopped"
        # Check if the job is pending
        else:
            return "pending"

    # Find all folders in the worker output directory
    job_folders = [f for f in WORKER_OUTPUT_DIR.iterdir() if f.is_dir()]
    # Get the status of each job
    status = {}
    running_worker_names = get_worker_names()
    for equation in job_folders:
        for algorithm in equation.iterdir():
            for trial in algorithm.iterdir():
                # Get the status of the job
                status[(equation.name, algorithm.name, trial.name)] = tuple([get_status_of_job(equation.name, algorithm.name, trial.name, running_worker_names)])
                if status[(equation.name, algorithm.name, trial.name)] == ("finished"):
                    # read the parameters.json file. If it fails, continue but write a warning
                    try:
                        with open(trial / "parameters.json", "r") as f:
                            parameters = json.load(f)
                    except:
                        print(f"Could not read parameters.json for {equation.name}/{algorithm.name}/{trial.name}")
                        continue
                    # Add the parameters to the status dictionary
                    # try to read last_n, converged
                    try:
                        status[(equation.name, algorithm.name, trial.name)] += (parameters["last_n"], parameters["converged"])
                    except:
                        # if failed save last_n, converged as None
                        status[(equation.name, algorithm.name, trial.name)] += (None, None)
    return status


def write_job_dict() -> dict:
    """
    Writes the status of all jobs to the job_dict.pickle file. Returns the status dictionary.
    """
    status = get_status_of_jobs()
    # CURRENTLY NOT USED
    # with open(JOB_DICT, "wb") as f:
    #     pickle.dump(status, f)
    return status


def count_jobs(status: dict) -> dict:
    """Returns a dictionary of the number of jobs for each status. 'finished', 'running', 'stopped', 'pending'"""
    count = {"finished": 0, "running": 0, "stopped": 0, "pending": 0}
    for s in status.values():
        count[s[0]] += 1
    return count



if __name__ == "__main__":
    status = get_status_of_jobs()
    print(len(status))
    status = write_job_dict()
    print(len(status))
    print(count_jobs(status))
    