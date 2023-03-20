import concurrent.futures
import os
from pathlib import Path
import argparse

# imports from JobOrchestration
from constants import SYREAL_PKG_DIR, LOSS_THRESHOLD, WORKER_OUTPUT_DIR
from worker import WorkerManager, Worker, load_worker_manager
from utils import Logger
from feynman import Feynman

# imports from syreal
import sys
parent_dir = SYREAL_PKG_DIR / "syreal"
sys.path.insert(0, str(parent_dir))

from syreal import Search
from default_kwargs import default_syreal_kwargs, default_pysr_kwargs
from jobs import feynman
from jobs.read_config import read_config

logger, keepfds = Logger(add_handler=False)

# Get worker from command line
parser = argparse.ArgumentParser()
parser.add_argument("--worker_name", type=str, default=0)
args = parser.parse_args()
worker_name = args.worker_name
# load worker manager from a pickle file
WorkerManager = load_worker_manager()
try:
    worker = WorkerManager.get_worker(worker_name)
except:
    logger.error(f"WORKER ERROR: Worker {worker_name} not found in WorkerManager.")
    raise Exception(f"WORKER ERROR: Worker {worker_name} not found in WorkerManager.")

# Get jobs from worker
jobs = worker.jobs

# read package config.yml
pkg_config = read_config()

MAX_WORKERS = len(os.sched_getaffinity(0))
OUTPUT_DIR = Path(pkg_config["directory"]["outputParentDirectory"])

#problem = feynman.mk_problems(Filter=dict(Filename="I.6.2b"))[0] # returns a list of problems which fulfill the Filter. In this case the list has only one element

def func(i):
    job = jobs[i]
    problem = Feynman(job['equation'])
    # TODO: add custom pysr_kwargs
    pysr_kwargs=dict(procs=0)
    default_pysr_kwargs.update(pysr_kwargs)
    kwargs = dict(
        algorithm=job['algorithm'],
        eq=problem.equation,
        boundaries=problem.boundaries,
        N_stop=problem.datapoints+100,
        # TODO: make datapoints better
        # TODO: declare warm start when status is 'stopped'
        seed=int(job['trial']),
        parentdir=str(WORKER_OUTPUT_DIR / job['equation'] / job['algorithm'] / job['trial']),
        pysr_params=default_pysr_kwargs,
        abs_loss_zero_tol=LOSS_THRESHOLD,
        warm_start=job['status'] == 'stopped',
    )
    default_syreal_kwargs.update(kwargs)
    Search(**default_syreal_kwargs)

executor = concurrent.futures.ProcessPoolExecutor(max_workers=len(jobs))
futures = [executor.submit(func, i) for i in range(len(jobs))]

for future in concurrent.futures.as_completed(futures):
    future.result()

executor.shutdown(wait=True)