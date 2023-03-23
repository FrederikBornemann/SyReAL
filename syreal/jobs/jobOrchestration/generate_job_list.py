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

from constants import FEYNMAN_CSV_FILE, JOB_LIST
import csv
import json
import numpy as np

def import_feynman_csv(ignore_equations=[]):
    with open(FEYNMAN_CSV_FILE, "r") as f:
        reader = csv.reader(f)
        equations = [row[0] for row in reader]
    # remove header
    equations = equations[1:]
    # remove ignored equations
    for eq in ignore_equations:
        equations.remove(eq)
    return equations

def generate_job_list(algorithms, trials, pysr_kwargs, kwargs, ignore_equations=[], equations=[]):
    if not equations:
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
    with open(JOB_LIST, "w") as f:
        json.dump(job_list, f, indent=4)


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

if __name__ == "__main__":
    #algorithms = ["random", "combinatory", "std", "complexity-std", "loss-std", "true-confusion"]
    algorithms = ["random", "combinatory"]
    
    base_range_growing = np.array([5, 10, 15, 20, 30, 40, 50, 60, 80, 100], dtype=int)
    base_range_linear = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=int)
    
    # make a grid of populations and iterations as pysr_kwargs
    grid_values = dict(populations=base_range_linear//2, niterations=base_range_growing)
    pysr_kwargs = make_parameters_grid(grid_values)
    #print(pysr_kwargs, len(pysr_kwargs))
    #pysr_kwargs = [{"populations":i, "niterations": i} for i in range(len(equations)))]
    kwargs = [{} for i in range(len(pysr_kwargs))]
    equations = []
    equations=["I.6.2b" for i in range(len(pysr_kwargs))]
    ignore_equations = []
    generate_job_list(algorithms, 15, pysr_kwargs, kwargs, ignore_equations, equations=equations)





