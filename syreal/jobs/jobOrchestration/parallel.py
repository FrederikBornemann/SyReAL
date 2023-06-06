# fmt: off
import concurrent.futures
import argparse
import json

# imports from JobOrchestration
from constants import SYREAL_PKG_DIR, LOSS_THRESHOLD, WORKER_OUTPUT_DIR, MAX_ITERATIONS, PYSR_PARAMETERS, SYREAL_PARAMETERS, BATCH_DIR, BATCH_SIZE
from worker import WorkerManager, Worker, load_worker_manager
from utils import Logger
from feynman import Feynman
from job_list_handler import write_job_list

# imports from syreal
import sys
parent_dir = SYREAL_PKG_DIR / "syreal"
sys.path.insert(0, str(parent_dir))
from default_kwargs import default_syreal_kwargs, default_pysr_kwargs
from syreal import Search

# Set up logging
logger, keepfds = Logger(add_handler=False)

# Get worker from command line
parser = argparse.ArgumentParser()
parser.add_argument("--worker_name", type=str, default=0)
args = parser.parse_args()
worker_name = args.worker_name
# load worker manager from a pickle file
WorkerManager = load_worker_manager()
# get the worker
try:
    worker = WorkerManager.get_worker(worker_name)
except:
    logger.error(
        f"WORKER ERROR: Worker {worker_name} not found in WorkerManager.")
    raise Exception(
        f"WORKER ERROR: Worker {worker_name} not found in WorkerManager.")

# Get jobs from worker
jobs = worker.jobs

# define function to run the jobs
def func(i):
    job = list(jobs[i][0])
    equation, algorithm, trial = job
    problem = Feynman(equation.split(" ")[1])
    # TODO: add custom pysr_kwargs
    pysr_kwargs = PYSR_PARAMETERS if PYSR_PARAMETERS else {}
    pysr_kwargs.update(dict(procs=0))
    default_pysr_kwargs.update(pysr_kwargs)
    kwargs = SYREAL_PARAMETERS if SYREAL_PARAMETERS else {}
    kwargs.update(dict(
        algorithm=algorithm,
        eq=problem.equation,
        boundaries=problem.boundaries,
        N_stop=MAX_ITERATIONS,
        # TODO: make datapoints better by using dynamic datapoints.
        seed=int(trial),
        parentdir=str(WORKER_OUTPUT_DIR / \
                      equation / algorithm / trial),
        pysr_params=default_pysr_kwargs,
        abs_loss_zero_tol=LOSS_THRESHOLD,
        warm_start=int(jobs[i][1]) == 0,
    ))
    default_syreal_kwargs.update(kwargs)
    # write Running.txt file with worker_name
    with open(WORKER_OUTPUT_DIR / equation / algorithm / trial / "Running.txt", "w") as f:
        f.write(worker_name)
    converged, iterations, exec_time = Search(**default_syreal_kwargs)
    # write parameters.json file and Done.txt file
    parameters = dict(
        converged=converged,
        last_n=iterations,
        exec_time=exec_time,)
    with open(WORKER_OUTPUT_DIR / equation / algorithm / trial / "parameters.json", "w") as f:
        json.dump(parameters, f)
    with open(WORKER_OUTPUT_DIR / equation / algorithm / trial / "Done.txt", "w") as f:
        f.write("Done")

def batch_jobs() -> list:
    """Load a batch of jobs from a pickle file and delete the file."""
    import os
    import pickle
    import random
    # Get a list of all pickle files in the directory
    pickle_files = [f for f in os.listdir(BATCH_DIR) if f.endswith('.pickle')]
    if len(pickle_files) == 0:
        # Return None if no pickle files are found
        return None
    # Select a random pickle file from the list
    pickle_file = random.choice(pickle_files)
    file_path = os.path.join(BATCH_DIR, pickle_file)
    # Read the pickle file
    with open(file_path, 'rb') as f:
        jobs = pickle.load(f)
    # Delete the pickle file
    os.remove(file_path)
    # Return the file path and the loaded data
    return jobs



MAX_PROCS = len(jobs)
with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCS) as executor:
    # run the jobs in parallel
    futures = [executor.submit(func, i) for i in range(len(jobs))]
    ACTIVE_PROCS = MAX_PROCS
    #logger.debug(f"WORKER {worker_name}: {executor._pending_work_items} with type {type(executor._pending_work_items)}")


    for future in concurrent.futures.as_completed(futures):
        future.result()
        ACTIVE_PROCS -= 1

        # If enough processes have completed, start a new batch
        if MAX_PROCS - ACTIVE_PROCS >= BATCH_SIZE:
            logger.debug(f"WORKER {worker_name}: Starting new batch of jobs")
            jobs = batch_jobs()
            if jobs is None:
                continue
            futures += [executor.submit(func, i) for i in range(len(jobs))]
            ACTIVE_PROCS += len(jobs)
    executor.shutdown(wait=True)

# min_running = int(len(futures) * 0.65)
# running_count = len(futures)
# # wait for all the jobs to finish
# for future in concurrent.futures.as_completed(futures):
#     future.result()
#     # Reason behind this is that the jobs take the same time if no other jobs are running. So it is better to have more jobs running at the same time.
#     running_count -= 1
#     if running_count <= min_running:
#         logger.info(
#             f"WORKER INFO: {worker_name} has {running_count} jobs running. Which is under the recuired job count. Cancelling remaining jobs.")
#         # cancel any remaining futures
#         for remaining_future in futures:
#             if remaining_future.running():
#                 remaining_future.cancel()
#         # shutdown the executor
#         result = worker.stop()
#         # interesting to see if this gets printed, because the worker is stopped before this.
#         if result:
#             logger.info(f"WORKER INFO: {worker_name} stopped successfully.")
#         else:
#             logger.error(f"WORKER ERROR: {worker_name} failed to stop.")
#         executor.shutdown(wait=False, cancel_futures=True)
#         break
# executor.shutdown(wait=True)
# # All the jobs are done now

# fmt: on