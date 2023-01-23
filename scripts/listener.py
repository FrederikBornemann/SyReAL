# listener.py

# Jobs:
# - (ead monitor + decide on seed+algo (prefer:  seed="running"->algo="stopped"  THEN  seed="running"->algo="pending"  THEN  seed="pending"->algo="pending"))
#   - (refer running random on one node! (helps with time difference of execution) (Maybe: If random is selected other procs on node run same algo for different seeds. Except when random is "stopped"))
# - parse seed,algo,info from distributor.py
# - start search with seed+algo (if "stopped" -> read data (if available) and resume search)
# - write algo "finnished" when done -> nrp -= 1

import sys
import os
import argparse
import watcher
import json
import pandas as pd
import sympy
from PySR_Search import Search
from read_config import read_config

config = read_config()

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--algorithm', type=str, required=True)
parser.add_argument('--info', type=str, required=True)
parser.add_argument('--parentdir', type=str, required=True)
parser.add_argument('--monitor', type=str, required=True)
parser.add_argument('--boundaries', type=json.loads, required=False) #True
parser.add_argument('--equation', type=sympy.sympify, required=True) 
parser.add_argument('--datapoint_exp', action='store_true')

# Parse the argument
Args = parser.parse_args()

seed = Args.seed
algo = Args.algorithm
info = Args.info
parentdir = Args.parentdir
monitor_path = Args.monitor
#boundaries = {var_name: [float(limits.split(":")[0]), float(limits.split(":")[1])] for var_name, limits in Args.boundaries.items()}
boundaries = {"sigma":[1.0,3.0],"theta":[1.0,3.0],"theta1":[1.0,3.0]}
#eq = Args.equation
datapoint_exp = Args.datapoint_exp
# Nstop
if datapoint_exp:
    N_stop = 2000
    
N_stop = 350
step_map = {0:1}
eq = "exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)".replace("pi", "3.1415")

args = dict(
    eq=eq,
    upper_sigma=0.0, 
    niterations=30, 
    boundaries=boundaries,
    parentdir=f"{parentdir}/{seed}",
    unary_operators=["neg","square","cube","exp","sqrt","sin","cos","tanh"], 
    binary_operators=["plus","sub","mult","div"],
    check_if_loss_zero=True,
    N_stop=N_stop,
    N_start = 5,
    seed=seed,
    equation_tracking=False,
    early_stop=True,
    step_multiplier=step_map,
    pysr_params=dict(populations=20, procs=20),
    warm_start = True if info == "stopped" else False,
    abs_loss_zero_tol=1e-6,
)
Search(algorithm=algo, **args)


# mark as finished 
with open(monitor_path, 'r') as f:
    monitor = json.load(f)
monitor["seeds"][str(seed)]["progress"][algo] = "finished"
with open(monitor_path, 'w') as f:
    json.dump(monitor, f)