# prio: r after f: s=0, p=1
#       r after r: s=2, p=3 (+2)
#       p after f: s=4, p=5 (+4)
#       p after r: s=6, p=7 (+6)
#       r after p: s=2, p=3 (+8)
#       p after p: s=6, p=7 (+8)

import pandas as pd
import numpy as np

def make_job_list(monitor, node_num_to_fill):
    seeds = monitor["seeds"]
    jobs_df = pd.DataFrame(columns=["prio", "seed", "algo", "info"])
    prev_status = "finished"
    for seed in seeds.items():
        status = seed[1]["status"]
        progress = seed[1]["progress"].items()
        add = 2 if prev_status == "running" else (4 if prev_status == "pending" else 0)
        if status == "finished":
            prev_status = status
            continue
        elif status == "running":
            add += 4 if prev_status == "pending" else 0
            for algo in progress:
                if algo[1] == "finished":
                    continue
                if algo[1] == "stopped":
                    jobs_df = pd.concat([jobs_df, pd.DataFrame.from_dict({"prio": 0+add, "seed": [seed[0]], "algo": algo[0], "Info": "stopped"})], ignore_index=True)
                elif algo[1] == "pending":
                    jobs_df = pd.concat([jobs_df, pd.DataFrame.from_dict({"prio": 1+add, "seed": [seed[0]], "algo": algo[0], "Info": "pending"})], ignore_index=True)
        elif status == "pending":
            add += 4
            for algo in progress:
                if algo[1] == "stopped":
                    jobs_df = pd.concat([jobs_df, pd.DataFrame.from_dict({"prio": 0+add, "seed": [seed[0]], "algo": algo[0], "Info": "stopped"})], ignore_index=True)
                elif algo[1] == "pending":
                    jobs_df = pd.concat([jobs_df, pd.DataFrame.from_dict({"prio": 1+add, "seed": [seed[0]], "algo": algo[0], "Info": "pending"})], ignore_index=True)
        prev_status = status
        if jobs_df.shape[0] > node_num_to_fill * 3 + 1:
            break
    jobs_df = jobs_df.sort_values(by=['prio']).reset_index(drop=True)
    jobs = list()
    for i, job in jobs_df.groupby(np.arange(len(jobs_df)) // 3):
        jobs.append(dict(seed=job.seed.tolist(), algorithm=job.algo.tolist(), info=job.Info.tolist()))
    return jobs[:node_num_to_fill]