# watcher.py

# Jobs:
# - modify algos "running" to "stopped" when last time modified > delta_time (10min)
#   - number (of) running processes (nrp) -= 1
#   - if nrp < process_num (2 processes per node. So maybe 10 nodes = 20 processes) -> call distributor.py (reads monitor.json and sends recuired number of jobs)
# - modify seed "finnished" when all algos have "finnished" -> progress_num += 1
# - modify seed "running" when an algo has "running"
# - create job list for prefered seeds+algos (seeds, algos, infos) (seeds: [0,0,0], algos=[std, random, combinatory], info=[pending, stopped, pending])

import distributor
import subprocess
import json
from job_list import make_job_list
import time
import datetime
import pandas as pd

import sys, os
# initial directory
cwd = os.getcwd()
# directory of "Output" folder
d = '/home/bornemaf/'
# trying to insert to directory
try:
    print("Inserting inside-", os.getcwd())
    os.chdir(d)      
except:
    print("Something wrong with specified directory. Exception- ")
    print(sys.exc_info())

# Old paths
algos = ["random", "combinatory", "std", "complexity-std", "loss-std", "true-confusion"]
seed_num = [0, 100]
procs_num = 30
node_num_desired = 10
node_num = len(str(subprocess.Popen( ['squeue','-u','bornemaf','-p','all'], stdout=subprocess.PIPE ).communicate()[0]).split(r'\n')[1:-1])
sleep_time = 120   # in seconds

def make_dir_names(filename, basepath=f"{os.getcwd()}/Output/Feynman_Trials"):
    parentdir = f"{basepath}/{filename}"
    dirs = dict(
        monitor=parentdir+"/monitor.json",
        data=parentdir+"/data",
        temp_scripts=parentdir,
        )
    return basepath, parentdir, dirs

def last_modified(path):
    try:
        file_mod_time = os.stat(path).st_mtime
        # Time in minutes since last modification of file
        last_time = (time.time() - file_mod_time) / 60
        return last_time
    except FileNotFoundError:
        return 99999.0

def update_monitor(path, node_num, jobs=None):
    with open(dirs["monitor"], 'r') as f:
        monitor = json.load(f)
    monitor["running"] = f"{node_num}/{monitor['running'].split('/')[1]}"
    if jobs == None:
        for seed in monitor["seeds"].items():
            if seed[1]["status"] == "running":
                # if seed was "running" and all algos are "finished", mark seed as "finished"
                if set(seed[1]["progress"].values()) == {'finished'}:
                    monitor["seeds"][seed[0]]["status"] = "finished"
                    monitor["progress"] = str(int(monitor["progress"].split("/")[0])+1)+"/"+monitor["progress"].split("/")[1]
                    continue
                for algo in seed[1]["progress"].items():
                    if algo[1] == "stopped":
                        try:
                            # if "stopped" is actually "finished"
                            with open(f'{dirs["data"]}/{seed[0]}/{algo[0]}/parameters.json', 'r') as f:
                                parameters = json.load(f)
                            if parameters["converged"] == True or (parameters["last_n"] == parameters["N_stop"]-1):
                                monitor["seeds"][seed[0]]["progress"][algo[0]] = "finished"
                        except:
                            pass
                    if algo[1] == "running" and last_modified(dirs["data"]+f"/{seed[0]}/{algo[0]}/parameters.json") > 15.0:
                        monitor["seeds"][seed[0]]["progress"][algo[0]] = "stopped"
                        try:
                            # if "running" is actually "finished"
                            with open(f'{dirs["data"]}/{seed[0]}/{algo[0]}/parameters.json', 'r') as f:
                                parameters = json.load(f)
                            if parameters["converged"] == True or (parameters["last_n"] == parameters["N_stop"]-1):
                                monitor["seeds"][seed[0]]["progress"][algo[0]] = "finished"
                        except:
                            pass                     
    else:
        for job in jobs:
            seeds, algos = job["seed"], job["algorithm"]
            for seed, algo in zip(seeds, algos):
                monitor["seeds"][str(seed)]["status"] = "running"
                monitor["seeds"][str(seed)]["progress"][algo] = "running"
    with open(dirs["monitor"], 'w') as f:
        json.dump(monitor, f)
    return monitor


if __name__ == "__main__":
    overall_finished = False
    while not overall_finished:
        # Make the main output directory
        try:
            os.makedirs(f"{os.getcwd()}/Output/Feynman_Trials") 
        except OSError as error: 
            print("GOOD ERROR \n" + str(error))
        # Import all feynman equations from the PySR github repo
        url = 'https://raw.githubusercontent.com/MilesCranmer/PySR/master/datasets/FeynmanEquations.csv'
        feynman_df = pd.read_csv(url)
        Filenames = feynman_df['Filename'].tolist()
        basepath, parentdir, dirs = make_dir_names(filename="")
        # If there's no main monitor json file, make one
        if not os.path.exists(basepath + "/main_monitor.json"):
            with open(basepath + "/main_monitor.json", 'w') as f:
                info = lambda x: dict(status="pending", progress="0", info=dict(equation=x.Formula.iloc[0], datapoints=None, step_map=None, hard_stop=None))
                data=dict(equations={i : info(feynman_df.loc[feynman_df['Filename'] == i]) for i in Filenames}, progress=f"0/{len(Filenames)}")
                json.dump(data, f)
                # Start the first equation (or filename)
                equation_to_start = Filenames[0]
        # Else, just read the existing one. In there, there are all the information about the current status of each feynman equations trials
        else:
            with open(basepath + "/main_monitor.json", 'r') as f:
                main_monitor = json.load(f)
                status = [x[1]["status"] for x in list(main_monitor["equations"].items())]
                # If all equations are finished: Terminate loop
                if set(status) == {"finished"}:
                    overall_finished = True
                    break
                if "finished" not in status:
                    equation_to_start = Filenames[0]
                # Start the equation (or filename) that comes after the last one that is "finished"
                equation_to_start = Filenames[next(i for i, x in enumerate(status) if x != "finished")] # Takes the first filename that is not finished
                datapoints_experiment = True if main_monitor["equations"][equation_to_start]["info"]["datapoints"] == None else False
        
        
        # TO BE DELETED: for debugginng purposes:
        equation_to_start = "I.6.2b"
        datapoints_experiment = False
        
        
        
        # Define directories where the trials need to run
        basepath, parentdir, dirs = make_dir_names(equation_to_start)
        # If no directory, make one
        try:
            os.makedirs(parentdir+"/data") 
        except OSError as error: 
            print("GOOD ERROR \n" + str(error)) 
        # Make monitor.json
        if not os.path.exists(dirs["monitor"]):
            with open(dirs["monitor"], 'w') as f:
                info = dict(status="pending", progress={algo : "pending" for algo in algos})
                data=dict(seeds={i : info for i in range(seed_num[0], seed_num[1])}, progress=f"0/{seed_num[1]-seed_num[0]}", running=f"0/{node_num_desired}")
                json.dump(data, f)
        # Read equation, boundaries, expected-number-of-datapoints, operators...?
        equation_row = feynman_df.loc[feynman_df['Filename'] == equation_to_start]
        equation = equation_row.Formula.iloc[0]
        boundaries = {equation_row[f"v{i}_name"].iloc[0]: f'{float(equation_row[f"v{i}_low"])}:{float(equation_row[f"v{i}_high"])}' for i in range(1, int(equation_row["# variables"]+1))} # Structure: {"var_name": "low:high", ...}
        # Enter trials-loop. This will run until all trials for this equation are finished
        while datapoints_experiment:
            node_num = len(str(subprocess.Popen( ['squeue','-u','bornemaf','-p','allcpu'], stdout=subprocess.PIPE ).communicate()[0]).split(r'\n')[1:-1])
            
        equation_finished = False
        while not equation_finished:
            # get number of used nodes
            node_num = len(str(subprocess.Popen( ['squeue','-u','bornemaf','-p','allcpu'], stdout=subprocess.PIPE ).communicate()[0]).split(r'\n')[1:-1])
            monitor = update_monitor(path=dirs["monitor"], node_num=node_num)
            # read all status from each seed. If all are finished -> mark equation as finished and end loop
            status_set = set([seed["status"] for seed in monitor["seeds"].values()])
            if status_set == {"finished"}:
                equation_finished = True
                break
            # run jobs if jobs are finished or stopped (update monitor)
            if node_num < node_num_desired:
                jobs = make_job_list(monitor, node_num_desired-node_num)
                print(jobs)
                distributor.Distributor(jobs, parentdir, dirs, boundaries, equation)
                update_monitor(path=dirs["monitor"], node_num=node_num, jobs=jobs)

            time.sleep(sleep_time)
        print("Done!")    
        break #To be deleted
        overall_finished = True if set(eq["status"] for eq in main_monitor["equations"].values()) == {"finished"} else False