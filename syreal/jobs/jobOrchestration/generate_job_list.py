# The structure of the job_list.json file is as follows:
# {
#     "equations": {
#         "equation_name": {
#             "parameters": {
#                 "pysr_kwargs": {
#                     "max_iterations": 100,
#                 },
#                 "kwargs": {
#                     "max_iterations": 100,
#                 },
#             "status": "running",
#             "trials": {
#                 "algorithm_name": {
#                     "trial_number": {
#                         "status": "pending",
#                         "converged": False,
#                         "iterations": 0,
#                         "exec_time": 0,
#                     },
#                 },
#             },
#         },
#     },
# }

from constants import FEYNMAN_CSV_FILE, JOB_LIST, WORKER_OUTPUT_DIR
import csv
import json
import numpy as np
from pathlib import Path

def import_feynman_csv(ignore_equations=[], order_by_datapoints=False):
    with open(FEYNMAN_CSV_FILE, "r") as f:
        reader = csv.reader(f)
        equations = []
        datapoints = []
        for row in reader:
            equations.append(row[0])
            datapoints.append(row[1])
    # remove header
    equations = equations[1:]
    datapoints = datapoints[1:]
    # order by datapoints if requested. So that the equations with the least datapoints are first
    if order_by_datapoints:
        equations = np.array(equations)
        datapoints = np.array(datapoints)
        datapoints = datapoints.astype(np.float)
        equations = equations[np.argsort(datapoints)]
    # remove ignored equations
    for eq in ignore_equations:
        equations.remove(eq)
    # make equations python list
    equations = list(equations)
    return equations

def read_worker_output_dir(job_list):
    for eq in job_list["equations"]:
        for algo in job_list["equations"][eq]["trials"]:
            for trial in job_list["equations"][eq]["trials"][algo]:
                try:
                    with open(WORKER_OUTPUT_DIR / eq / algo / str(trial) / f"parameters.json", "r") as f:
                        params = json.load(f)
                except:
                    continue
                else:
                    try:
                        params["converged"]
                    except:
                        job_list["equations"][eq]["trials"][algo][trial]["status"] = "stopped"
                    else:
                        if params["converged"]:
                            job_list["equations"][eq]["trials"][algo][trial]["status"] = "finished"
                            job_list["equations"][eq]["trials"][algo][trial]["converged"] = params["converged"]
                            job_list["equations"][eq]["trials"][algo][trial]["iterations"] = params["last_n"]
                            job_list["equations"][eq]["trials"][algo][trial]["exec_time"] = params["time"]["Execution time"]                     
    return job_list

def generate_job_list(algorithms, trials, pysr_kwargs, kwargs, ignore_equations=[], equations=[], job_list_dir=None):
    """
    Generates a job list for the given algorithms and trials. The job list is saved to the job_list.json file.

    algorithms : list
        List of algorithms to run.
    trials : int
        Number of trials to run for each algorithm.
    pysr_kwargs : dict
        Parameters to pass to PySR.
    kwargs : dict
        Parameters to pass to the algorithm.
    ignore_equations : list, optional
        List of equations to ignore. The default is [].
    equations : list, optional
        List of equations to run. The default is [], in which case all equations are run.

    """
    # if equations is empty numpy array
    if not list(equations):
        equations = import_feynman_csv(ignore_equations)
    if not isinstance(pysr_kwargs, list):
        pysr_kwargs = [pysr_kwargs for i in range(len(equations))]
    if not isinstance(kwargs, list):
        kwargs = [kwargs for i in range(len(equations))]
    job_list = {
        "equations": {
            f"{i} {eq}": {
                "parameters": {
                    "pysr_kwargs": pysr_kwargs[i],
                    "kwargs": kwargs[i],
                },
                "status": "pending",
                "trials": {
                    algo: {
                        str(trial): {
                            "status": "pending",
                            "converged": False,
                            "iterations": 0,
                            "exec_time": 0,
                        }
                        for trial in range(trials)
                    }
                    for algo in algorithms
                },
            }
            for i, eq in enumerate(equations)
        },
    }
    # Read the worker output directory to update the status of the jobs
    job_list_file = JOB_LIST if job_list_dir is None else Path(job_list_dir) / "job_list.json"
    job_list = read_worker_output_dir(job_list)
    # Create the directory for the job list
    job_list_dir = job_list_file.parent
    job_list_dir.mkdir(parents=True, exist_ok=True)
    # Save the job list to the job_list.json file
    with open(job_list_file, "w") as f:
        json.dump(job_list, f)


def make_parameters_grid(parameters:dict[str, list]):
    import itertools
    # Compute Cartesian product of hyperparameters
    param_grid = list(itertools.product(*parameters.values()))
    # Convert to list of dictionaries
    grid_list = []
    for params in param_grid:
        # map params to int
        params = [int(p) for p in params]
        param_dict = dict(zip(parameters.keys(), params))
        grid_list.append(param_dict)

    return grid_list

def clear_backups():
    from constants import JOB_LIST_BACKUP_DIR
    # remove all files in JOB_LIST_BACKUP_DIR
    for file in JOB_LIST_BACKUP_DIR.iterdir():
        file.unlink()

def grid_search_job_list(algorithms, base_range_growing, base_range_linear):
    # make a grid of populations and iterations as pysr_kwargs
    grid_values = dict(populations=base_range_linear//2, niterations=base_range_growing)
    pysr_kwargs = make_parameters_grid(grid_values)
    kwargs = [{} for i in range(len(pysr_kwargs))]
    equations = []
    equations=["I.6.2b" for i in range(len(pysr_kwargs))]
    ignore_equations = []
    trials_per_algorithm = 15
    generate_job_list(algorithms, trials_per_algorithm, pysr_kwargs, kwargs, ignore_equations, equations)
    clear_backups()

if __name__ == "__main__":
    parts = 10
    job_list_dirs = [f"/beegfs/desy/user/bornemaf/data/syreal_output/Feynman_Experiment_1_part-{i}/" for i in range(parts)]
    algorithms = ["random", "combinatory", "std", "complexity-std", "loss-std", "true-confusion"]
    equations = import_feynman_csv(order_by_datapoints=True)
    # devide the equations into parts
    equations = np.array_split(equations, parts)
    pysr_kwargs = [{"populations": 35, "niterations": 20} for i in range(len(equations))]
    kwargs = [{} for i in range(len(equations))]
    trials_per_algorithm = 100
    for i, job_list_dir in enumerate(job_list_dirs):
        generate_job_list(algorithms, trials_per_algorithm, pysr_kwargs[i], kwargs[i], equations=equations[i], job_list_dir=job_list_dir)


