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
from prettytable import PrettyTable


def get_SLURM_monitor() -> pd.DataFrame:
    """Get the SLURM queue as a pandas dataframe."""
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
def check_SLURM_monitor(monitor) -> tuple[pd.DataFrame, bool, pd.DataFrame, dict]:
    """
    Check if the SLURM queue has changed. Returns a tuple with the following values:
    - monitor: the new SLURM queue as a pandas dataframe
    - changed: True if the SLURM queue has changed, False otherwise
    - changed_rows: the rows that have changed in the SLURM queue
    - updated_jobs: a dictionary with the names of the jobs that have changed as keys and the changes as values
    """
    new_monitor = get_SLURM_monitor()

    # drop all columns except jobid, state, and nodelist
    monitor = monitor[['jobid', 'name', 'state', 'nodelist']]
    new_monitor_reduced = new_monitor[['jobid', 'name', 'state', 'nodelist']]
    # compare the two monitors. If they are the same, return False. If they are different, return True
    changed_rows = monitor.merge(
        new_monitor_reduced, indicator=True, how='outer').loc[lambda x: x['_merge'] != 'both']
    alert = {}
    updated_jobs = {}
    if not changed_rows.empty:
        # get the rows that were deleted
        deleted_rows = changed_rows[changed_rows['_merge'] == 'left_only']
        # get the rows that were added
        added_rows = changed_rows[changed_rows['_merge'] == 'right_only']
        # combine the deleted and added rows for each name
        for name in changed_rows['name'].unique():
            alert[name] = {}
            try:
                alert[name]['deleted'] = deleted_rows[deleted_rows['name']
                                                      == name]['state'].values[0]
            except IndexError:
                alert[name]['deleted'] = ''
            try:
                alert[name]['added'] = added_rows[added_rows['name']
                                                  == name]['state'].values[0]
            except IndexError:
                alert[name]['added'] = ''
        updated_jobs = {}
        for name, items in alert.items():
            deleted, added = items.values()
            if (deleted == 'PD' or deleted == '') and added == 'R':
                updated_jobs[name] = 'started'
            elif deleted == 'R' and (added == 'CD' or added == 'CG'):
                updated_jobs[name] = 'completed'
            elif (deleted == 'R' or deleted == 'PD') and added == '':
                updated_jobs[name] = 'stopped'
            elif deleted == 'R' and added == 'F':
                updated_jobs[name] = 'failed'
            elif (deleted == 'R' or deleted == 'PD') and added == 'PR':
                updated_jobs[name] = 'preempted'
            elif deleted == '' and added == 'PD':
                updated_jobs[name] = 'queued'
    return new_monitor, not changed_rows.empty, changed_rows, updated_jobs

def get_worker_number() -> int:
    """Get the number of workers in the SLURM queue."""
    return get_SLURM_monitor().shape[0]

def get_worker_names(exclude_CG=False) -> list:
    """Get the names of the workers in the SLURM queue."""
    if exclude_CG:
        return get_SLURM_monitor()[get_SLURM_monitor()['state'] != 'CG']['name'].values
    return get_SLURM_monitor()['name'].values

def get_worker_ids(exclude_CG=False) -> list:
    """Get the ids of the workers in the SLURM queue."""
    if exclude_CG:
        return get_SLURM_monitor()[get_SLURM_monitor()['state'] != 'CG']['jobid'].values
    return get_SLURM_monitor()['jobid'].values


def print_jobs_as_table(job_df, alerts):
    """Print the jobs as a table."""
    table_data = job_df.values.tolist()
    # Create a table with columns
    table = PrettyTable()
    table.field_names = job_df.columns
    # Add the data to the table
    for row in table_data:
        table.add_row(row)
    # Define the colors for each state
    colors = {"R": "\033[92m", "CG": "\033[91m", "PD": "\033[93m", "CF": "\033[93m", "CD": "\033[93m", "F": "\033[93m",
              "TO": "\033[93m", "NF": "\033[93m", "RV": "\033[93m", "SE": "\033[93m", "S": "\033[93m", "UNK": "\033[93m"}

    colors_alert = {"started": "\033[92m", "completed": "\033[94m", "stopped": "\033[91m", "failed": "\033[91m", "preempted": "\033[91m", "queued": "\033[93m"}

    # Update the table with colored values
    # index of the state column
    state_index = job_df.columns.get_loc("state")
    for i in range(len(table._rows)):
        state = table._rows[i][state_index]
        table._rows[i][state_index] = f"{colors[state]}{state}\033[0m"
    # clear the console
    os.system('clear')

    # Print the table
    print(table)
    if len(alerts) > 0:
        for name, alert in alerts.items():
            print(f"\033[1m{name} is {colors_alert[alert]}{alert}\033[0m\033[0m")
