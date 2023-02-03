import concurrent.futures
from syreal.syreal import Search
from syreal.default_kwargs import default_syreal_kwargs
from syreal.jobs import feynman
import os

MAX_WORKERS = len(os.sched_getaffinity(0))

problem = feynman.mk_problem(Filter=dict(Filename="I.6.2b"))[0] # returns a list of problems which fulfill the Filter. In this case the list has only one element

def func(i):
    kwargs = dict(
        algorithm="std"
        eq=problem.form,
        boundaries=problem.boundaries,
        N_stop=problem.dp
        parentdir=
    )
    Search(**default_syreal_kwargs.update(kwargs))


with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(func, i) for i in range(MAX_WORKERS)]