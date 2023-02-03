import os
from syreal.syreal import Search
from syreal.jobs.read_config import read_config
#from syreal.jobs.feynman_loader import load_datapoints

config = read_config()

#dataset = load_datapoints(files=["I.6.2b"], inside=False)[0]

args = dict(
    eq="exp(-(theta/sigma)**2/2)/(sqrt(2*3.1415)*sigma)",
    upper_sigma=0.0, 
    niterations=30, 
    boundaries={"sigma":[1.0,3.0],"theta":[1.0,3.0]},
    parentdir=f"{config['directory']['outputParentDirectory']}/example_20-1",
    unary_operators=["neg","square","cube","exp","sqrt","sin","cos","tanh"], 
    binary_operators=["plus","sub","mult","div"],
    N_stop=30,
    N_start = 5,
    equation_tracking=False,
    early_stop=False,
    step_multiplier={0:1},
    pysr_params=dict(populations=20, procs=1),
    warm_start = False,
    check_if_loss_zero=True,
    abs_loss_zero_tol=1e-6,
    generative=True,
    dataset=None,
    export_confusion_score=True,
)

Search(algorithm="std", **args)
