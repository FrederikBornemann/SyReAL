import numpy as np
import pandas as pd
from scipy import stats
import json
from job_list_handler import read_job_list
from priority_queue import flatten_job_list
from constants import TEMP_DIR, WORKER_OUTPUT_DIR


def get_finished_jobs():
    job_list = read_job_list()
    flatten_jobs = list(flatten_job_list(
        job_list, skip_finished=False, skip_running=True))
    finished_jobs = [job for job in flatten_jobs if list(
        job.values())[0]["status"] == "finished"]
    return finished_jobs


def get_parameters_json():
    """Returns a dictionary of the parameters.json file which is in the file of each job."""
    finished_jobs = get_finished_jobs()
    parameters_json = []
    for job in finished_jobs:
        # read the parameters.json file
        job_dict = list(job.values())[0]
        equation = job_dict["equation"]
        algorithm = job_dict["algorithm"]
        trial = job_dict["trial"]
        try:
            with open(WORKER_OUTPUT_DIR / equation / algorithm / str(trial) / "parameters.json", "r") as f:
                parameters_json.append((job, json.load(f)))
        except Exception:
            print(
                f"Could not read parameters.json for {equation}/{algorithm}/{trial}")
    return parameters_json


def get_finished_jobs_summary(export_csv=False):
    finished_jobs = get_finished_jobs()
    # for each equation, get the mean iterations, mean time, etc.
    eq_dict = {}
    for job in finished_jobs:
        job_dict = list(job.values())[0]
        eq = job_dict["equation"]
        if eq not in eq_dict:
            eq_dict[eq] = []
        eq_dict[eq].append(job_dict)
    # get the mean iterations, mean time, etc.
    eq_summary = {}
    for eq in eq_dict:
        eq_summary[eq] = {}
        eq_summary[eq][("random", "num_trials")] = len(
            [job for job in eq_dict[eq] if job["algorithm"] == "random"])
        eq_summary[eq][("random", "mean_steps")] = np.mean(
            [job["iterations"] for job in eq_dict[eq] if job["algorithm"] == "random"])
        eq_summary[eq][("random", "std_iterations")] = np.std(
            [job["iterations"] for job in eq_dict[eq] if job["algorithm"] == "random"])
        # convert the time to seconds. Format is "HH:MM:SS"
        eq_summary[eq][("random", "mean_time")] = stats.trim_mean([int(time.split(":")[0]) * 60 + int(time.split(":")[1]) * 1 + int(
            time.split(":")[2]) // 60 for time in [job["exec_time"] for job in eq_dict[eq] if job["algorithm"] == "random"]], 0.1)
        eq_summary[eq][("random", "std_time")] = np.std([int(time.split(":")[0]) * 60 + int(time.split(":")[1]) * 1 + int(
            time.split(":")[2]) // 60 for time in [job["exec_time"] for job in eq_dict[eq] if job["algorithm"] == "random"]])

        # for algorithm = "combinatory"
        eq_summary[eq][("combinatory", "num_trials")] = len(
            [job for job in eq_dict[eq] if job["algorithm"] == "combinatory"])
        eq_summary[eq][("combinatory", "mean_steps")] = np.mean(
            [job["iterations"] for job in eq_dict[eq] if job["algorithm"] == "combinatory"])
        eq_summary[eq][("combinatory", "std_iterations")] = np.std(
            [job["iterations"] for job in eq_dict[eq] if job["algorithm"] == "combinatory"])
        # convert the time to seconds. Format is "HH:MM:SS"
        eq_summary[eq][("combinatory", "mean_time")] = stats.trim_mean([int(time.split(":")[0]) * 60 + int(time.split(":")[1]) * 1 + int(
            time.split(":")[2]) // 60 for time in [job["exec_time"] for job in eq_dict[eq] if job["algorithm"] == "combinatory"]], 0.1)
        eq_summary[eq][("combinatory", "std_time")] = np.std([int(time.split(":")[0]) * 60 + int(time.split(":")[1]) * 1 + int(
            time.split(":")[2]) // 60 for time in [job["exec_time"] for job in eq_dict[eq] if job["algorithm"] == "combinatory"]])
    # create a DataFrame
    df = pd.DataFrame.from_dict(eq_summary, orient="index")
    index = [eq_dict[eq][0]["equation_params"]["pysr_kwargs"]
             for eq in eq_dict]
    #index = [f"pop: {x['populations']}, iter: {x['niterations']}" for x in index]

    populations = [x["populations"] for x in index]
    niterations = [x["niterations"] for x in index]
    # Create the MultiIndex for rows
    idx = pd.MultiIndex.from_arrays([populations, niterations], names=[
                                    "populations", "niterations"])
    df.index = idx
    # add a top-level index for the column names
    df.columns = pd.MultiIndex.from_tuples(
        df.columns, names=["algorithm", "summary"])
    if export_csv:
        df.to_csv(TEMP_DIR / "finished_jobs_summary.csv", index=True)
    return df


def get_finished_jobs_summary_2(export_csv=False):
    lst = get_parameters_json()
    finished_jobs = [x[0] for x in lst]
    parameters_json = [x[1] for x in lst]
    # for each equation, get the mean iterations, mean time, etc.
    eq_dict = {}
    for job, params in zip(finished_jobs, parameters_json):
        job_dict = list(job.values())[0]
        eq = job_dict["equation"]
        if eq not in eq_dict:
            eq_dict[eq] = []
        eq_dict[eq].append((job_dict, params))
    # get the mean iterations, mean time, etc.
    print(eq_dict['85 I.6.2b'][1][1]['time']['Second model fit time'])
    eq_summary = {}
    for i, eq in enumerate(eq_dict):
        for j, job in enumerate(eq_dict[eq]):
            try:
                job[1]["time"]["Second model fit time"]
            except:
                print(i, j, job[1]["time"])
                eq_dict[eq].pop(j)
        eq_summary[eq] = {}
        eq_summary[eq][("random", "num_trials")] = len(
            [job[0] for job in eq_dict[eq] if job[0]["algorithm"] == "random"])
        eq_summary[eq][("random", "mean_steps")] = stats.trim_mean(
            [job[0]["iterations"] for job in eq_dict[eq] if job[0]["algorithm"] == "random"], 0.1)
        eq_summary[eq][("random", "std_iterations")] = np.std(
            [job[0]["iterations"] for job in eq_dict[eq] if job[0]["algorithm"] == "random"])
        # convert the time to seconds. Format is "HH:MM:SS"
        eq_summary[eq][("random", "mean_time_per_step")] = stats.trim_mean([int(time.split(":")[0]) * 3600 + int(time.split(":")[1]) * 60 + int(
            time.split(":")[2]) * 1 for time in [job[1]["time"]["Second model fit time"] for job in eq_dict[eq] if job[0]["algorithm"] == "random"]], 0.1)
        eq_summary[eq][("random", "std_time")] = np.std([int(time.split(":")[0]) * 60 + int(time.split(":")[1]) * 1 + int(time.split(
            ":")[2]) // 60 for time in [job[1]["time"]["Second model fit time"] for job in eq_dict[eq] if job[0]["algorithm"] == "random"]])

        # for algorithm = "combinatory"
        eq_summary[eq][("combinatory", "num_trials")] = len(
            [job[0] for job in eq_dict[eq] if job[0]["algorithm"] == "combinatory"])
        eq_summary[eq][("combinatory", "mean_steps")] = stats.trim_mean(
            [job[0]["iterations"] for job in eq_dict[eq] if job[0]["algorithm"] == "combinatory"], 0.1)
        eq_summary[eq][("combinatory", "std_iterations")] = np.std(
            [job[0]["iterations"] for job in eq_dict[eq] if job[0]["algorithm"] == "combinatory"])
        # convert the time to seconds. Format is "HH:MM:SS"
        eq_summary[eq][("combinatory", "mean_time_per_step")] = stats.trim_mean([int(time.split(":")[0]) * 3600 + int(time.split(":")[1]) * 60 + int(
            time.split(":")[2]) * 1 for time in [job[1]["time"]["Second model fit time"] for job in eq_dict[eq] if job[0]["algorithm"] == "combinatory"]], 0.1)
        eq_summary[eq][("combinatory", "std_time")] = np.std([int(time.split(":")[0]) * 60 + int(time.split(":")[1]) * 1 + int(time.split(
            ":")[2]) // 60 for time in [job[1]["time"]["Second model fit time"] for job in eq_dict[eq] if job[0]["algorithm"] == "combinatory"]])
        print(f"Finished {i} equations")
    # create a DataFrame
    df = pd.DataFrame.from_dict(eq_summary, orient="index")
    index = [eq_dict[eq][0][0]["equation_params"]["pysr_kwargs"]
             for eq in eq_dict]
    #index = [f"pop: {x['populations']}, iter: {x['niterations']}" for x in index]

    populations = [x["populations"] for x in index]
    niterations = [x["niterations"] for x in index]
    # Create the MultiIndex for rows
    idx = pd.MultiIndex.from_arrays([populations, niterations], names=[
                                    "populations", "niterations"])
    df.index = idx
    # add a top-level index for the column names
    df.columns = pd.MultiIndex.from_tuples(
        df.columns, names=["algorithm", "summary"])
    if export_csv:
        df.to_csv(TEMP_DIR / "finished_jobs_summary_2.csv", index=True)
    return df


# print(get_finished_jobs_summary(export_csv=True))
print(get_finished_jobs_summary_2(export_csv=True))
