from syreal import Search
from default_kwargs import default_syreal_kwargs
from default_kwargs import default_pysr_kwargs

kwargs = dict()
kwargs.update(dict(
        algorithm="random",
        eq="x_1**2+x_2",
        boundaries={"x_1": [-1, 1], "x_2": [-1, 1]},
        N_stop=100,
        seed=1,
        parentdir="/beegfs/desy/user/bornemaf/data/SyReAL/syreal/testing/job_dir",
        warm_start=True,
        testing=True,
    ))
default_syreal_kwargs.update(kwargs)
converged, iterations, exec_time = Search(**default_syreal_kwargs)