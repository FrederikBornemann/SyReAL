import os
import sys
# import python files from a different directory
sys.path.insert(1, f'{os.getcwd()}/scripts')
from PySR_Search import Search
from read_config import read_config

config = read_config(inside=False)

args = dict(
    eq="",
    upper_sigma=0.0, 
    niterations=30, 
    boundaries={"sigma":[1.0,3.0],"theta":[1.0,3.0],"theta1":[1.0,3.0]},
    parentdir=f"{config['directory']['output-parent-directory']}/example",
    unary_operators=["neg","square","cube","exp","sqrt","sin","cos","tanh"], 
    binary_operators=["plus","sub","mult","div"],
    N_stop=15,
    N_start = 5,
    equation_tracking=False,
    early_stop=True,
    step_multiplier={0:1},
    pysr_params=dict(populations=20, procs=20),
    warm_start = False,
    check_if_loss_zero=True,
    abs_loss_zero_tol=1e-6,
)

Search(algorithm=algo, **args)
