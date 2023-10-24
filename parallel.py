import concurrent.futures
import os
from pathlib import Path
from syreal.syreal import Search
from syreal.default_kwargs import default_syreal_kwargs, default_pysr_kwargs
from syreal.jobs import feynman
from syreal.jobs.read_config import read_config

# TODO: read job ticket and create a jobs object, or import Worker
# TODO: run jobs in parallel, right now it is just a single equation


# read package config.yml
pkg_config = read_config()

MAX_WORKERS = len(os.sched_getaffinity(0))
OUTPUT_DIR = Path(pkg_config["directory"]["outputParentDirectory"])

problem = feynman.mk_problems(Filter=dict(Filename="I.6.2b"))[0] # returns a list of problems which fulfill the Filter. In this case the list has only one element

def func(i):
    pysr_kwargs=dict(procs=0)
    default_pysr_kwargs.update(pysr_kwargs)
    kwargs = dict(
        algorithm="std",
        eq=problem.form,
        boundaries=problem.boundaries,
        N_stop=problem.dp,
        parentdir=str(OUTPUT_DIR / "Parallel" / f"{i}" / "std"),
        pysr_params=default_pysr_kwargs,
    )
    default_syreal_kwargs.update(kwargs)
    Search(**default_syreal_kwargs)

executor = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)
futures = [executor.submit(func, i) for i in range(MAX_WORKERS)]

for future in concurrent.futures.as_completed(futures):
    future.result()

executor.shutdown(wait=True)