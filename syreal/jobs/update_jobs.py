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
        for job in monitor["jobs"].items():
            if job[1]["status"] == "running":
                # if job was "running" and all algos are "finished", mark job as "finished"
                if set(job[1]["progress"].values()) == {'finished'}:
                    monitor["jobs"][job[0]]["status"] = "finished"
                    monitor["progress"] = str(int(monitor["progress"].split("/")[0])+1)+"/"+monitor["progress"].split("/")[1]
                    continue
                for algo in job[1]["progress"].items():
                    if algo[1] == "stopped":
                        try:
                            # if "stopped" is actually "finished"
                            with open(f'{dirs["data"]}/{job[0]}/{algo[0]}/parameters.json', 'r') as f:
                                parameters = json.load(f)
                            if parameters["converged"] == True or (parameters["last_n"] == parameters["N_stop"]-1):
                                monitor["jobs"][job[0]]["progress"][algo[0]] = "finished"
                        except:
                            pass
                    if algo[1] == "running" and last_modified(dirs["data"]+f"/{job[0]}/{algo[0]}/parameters.json") > 15.0:
                        monitor["jobs"][job[0]]["progress"][algo[0]] = "stopped"
                        try:
                            # if "running" is actually "finished"
                            with open(f'{dirs["data"]}/{job[0]}/{algo[0]}/parameters.json', 'r') as f:
                                parameters = json.load(f)
                            if parameters["converged"] == True or (parameters["last_n"] == parameters["N_stop"]-1):
                                monitor["jobs"][job[0]]["progress"][algo[0]] = "finished"
                        except:
                            pass                     
    else:
        for job in jobs:
            jobs, algos = job["job"], job["algorithm"]
            for job, algo in zip(jobs, algos):
                monitor["jobs"][str(job)]["status"] = "running"
                monitor["jobs"][str(job)]["progress"][algo] = "running"
    with open(dirs["monitor"], 'w') as f:
        json.dump(monitor, f)
    return monitor