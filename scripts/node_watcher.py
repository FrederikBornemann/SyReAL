# node_watcher.py

# Jobs:
#  - watch the progress of its node jobs
#  - finishes if all processes are done

import time
import watcher
import argparse
import json

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--seeds', type=int, required=True, nargs='+', default=[])
parser.add_argument('--algorithms', type=str, required=True, nargs='+', default=[])
parser.add_argument('--monitor', type=str, required=True)
# Parse the argument
Args = parser.parse_args()

seeds = Args.seeds
algos = Args.algorithms
monitor_path = Args.monitor

while True:
    with open(monitor_path, 'r') as f:
        monitor = json.load(f)
    checklist = {f"{seed}_{algo}":False for seed, algo in zip(seeds, algos)}
    for seed, algo in zip(seeds, algos):
        if monitor["seeds"][str(seed)]["progress"][algo] == "finished":
            checklist[f"{seed}_{algo}"] = True
    if set(checklist.values()) == {True}:
        break
    time.sleep(60)
        