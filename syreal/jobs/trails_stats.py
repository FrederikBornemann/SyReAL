import json
import numpy as np
import os
import watcher

# Jobs:
#  - Go trough all seeds that are marked as "finished" and calc all needed stats
#  - Maybe make graphs


parentdir = watcher.parentdir
dirs = watcher.dirs

# read monitor json
with open(dirs["monitor"], 'r') as f:
    monitor = json.load(f)

last_n_list = list()
for seed in monitor["seeds"].items():
    seed_num = seed[0]
    seed_algos = seed[1]["progress"]
    if seed[1]["status"] == "finished":
        algo_dict = dict()
        for algo in seed_algos.values():
            path = f"{dirs["data"]}/{seed_num}/{algo}"
            with open(path+"/parameters.json", 'r') as f:
                parameters = json.load(f)
            algo_dict[algo] = parameters["last_n"]
        last_n_list.append(algo_dict)

for algo_dict in last_n_list:
    print(algo_dict)
            