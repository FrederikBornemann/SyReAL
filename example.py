import os
import sys
# import python files from a different directory
sys.path.insert(1, f'{os.getcwd()}/scripts')
from PySR_Search import Search
from read_config import read_config
from feynman_loader import load_datapoints

config = read_config(inside=False)

dataset = load_datapoints(files=["I.6.2b"], inside=False)[0]

args = dict(
    eq="",
    upper_sigma=0.0, 
    niterations=30, 
    boundaries={"sigma":[1.0,3.0],"theta":[1.0,3.0],"theta1":[1.0,3.0]},
    parentdir=f"{config['directory']['outputParentDirectory']}/example",
    unary_operators=["neg","square","cube","exp","sqrt","sin","cos","tanh"], 
    binary_operators=["plus","sub","mult","div"],
    N_stop=350,
    N_start = 5,
    equation_tracking=False,
    early_stop=True,
    step_multiplier={0:1},
    pysr_params=dict(populations=20, procs=20),
    warm_start = False,
    check_if_loss_zero=True,
    abs_loss_zero_tol=1e-6,
    generative=False,
    dataset=dataset,
)

Search(algorithm="std", **args)
