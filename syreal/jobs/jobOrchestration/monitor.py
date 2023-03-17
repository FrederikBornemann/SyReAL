# The monitor watches the SLURM queue and detects changes
# The monitor is a daemon that runs in the background
# The monitor is started by the job orchestrator
# The monitor is responsible for updating the job list

import os
import subprocess
import json
from pathlib import Path
import pandas as pd
import time


def get_SLURM_monitor():
    monitor = [x.split() for x in str(subprocess.Popen(['squeue', '-u', 'bornemaf',
                                                        '-p', 'allcpu'], stdout=subprocess.PIPE).communicate()[0]).split(r'\n')[1:-1]]
    if len(monitor) == 0:
        return pd.DataFrame(columns=['jobid', 'partition', 'name', 'user', 'state', 'time', 'nodes', 'nodelist'])
    # place every entry in the list into a dictionary with keys: jobid, partition, name, user, state, time, nodes, nodelist
    monitor = [dict(zip(['jobid', 'partition', 'name', 'user',
                    'state', 'time', 'nodes', 'nodelist'], x)) for x in monitor]
    # convert the jobid and nodes columns to integers
    for x in monitor:
        x['jobid'] = int(x['jobid'])
        x['nodes'] = int(x['nodes'])
    # convert monitor to pandas dataframe
    monitor = pd.DataFrame(monitor)
    # print(monitor)
    return monitor

# check if SLURM queue has changed and return the changed rows in the queue
def check_SLURM_monitor(monitor):
    new_monitor = get_SLURM_monitor()
    if new_monitor.empty:
        return new_monitor, False, pd.DataFrame()
    # drop all columns except jobid, state, and nodelist
    monitor = monitor[['jobid', 'name', 'state', 'nodelist']]
    new_monitor = new_monitor[['jobid', 'name', 'state', 'nodelist']]
    # compare the two monitors. If they are the same, return False. If they are different, return True
    changed_rows = monitor.merge(new_monitor, indicator=True, how='outer').loc[lambda x : x['_merge'] != 'both']
    
    return new_monitor, not changed_rows.empty, changed_rows


def loop_SLURM_monitor():
    monitor = get_SLURM_monitor()
    while True:
        monitor, changed, changed_rows = check_SLURM_monitor(monitor)
        if changed:
            print(changed_rows)
            # sort the changed rows by name. If two jobs have the same name
        print('.')
        time.sleep(3)



if __name__ == "__main__":
    loop_SLURM_monitor()
