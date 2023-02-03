pkg_config = read_config()

def get_number_of_used_nodes(partition='allcpu', maxwellUsername=pkg_config['user']['maxwellUsername']):
    return len(str(subprocess.Popen( ['squeue','-u', maxwellUsername,'-p', partition], stdout=subprocess.PIPE ).communicate()[0]).split(r'\n')[1:-1])

while True:
    # get number of used nodes
    node_num = get_number_of_used_nodes()
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