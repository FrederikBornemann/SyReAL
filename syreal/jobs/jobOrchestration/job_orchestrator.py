# TODO
# [x] read job list
# [x] make job tickets
# [] spawn jobs
# [] update job list


# This is the main job orchestrator. It is responsible for spawning jobs and updating the job list.
# It is also responsible for spawning the job monitor, which is responsible for monitoring the jobs

import time


from priority_queue import make_priority_queue
from utils import Logger
from monitor import get_worker_number
from worker import WorkerManager, Worker
from email_scraper import check_for_new_events
from job_list_handler import validate_jobs, read_job_list, clear_job_tickets, write_job_tickets


from constants import WORKER_NUM, PARALLEL_PROCS, SCHEDULER_INTERVAL

def start_job_orchestrator():
    import signal
    import sys

    logger, keep_fds = Logger(add_handler=False)

    def sigterm_handler(signum, frame):
        logger.critical(
            "Received TERMINATION signal. Exiting the job orchestrator killing all jobs...")
        # TODO: terminate the spawner, wait for the current job to finish, and then exit
        # save the last
        sys.exit(0)

    def sigusr1_handler(signum, frame):
        logger.critical(
            "Received EXITING signal. Exiting the job orchestrator but not killing any jobs...")
        # TODO: Kill all jobs, delete all job tickets, and mark all running jobs as stopped
        sys.exit(0)

    # Set the signal handler for SIGTERM and SIGKILL
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGUSR1, sigusr1_handler)

    # Start the job orchestrator
    logger.info("Job orchestrator has started.")


    # MAIN LOOP
    while True:
        start_time = time.time()
        # read the job list
        job_list = read_job_list()
        # check if there are any new events
        alerts = check_for_new_events()
        if alerts:
            for alert in alerts:
                if alert['status'] == 'COMPLETED':
                    # remove the job ticket from the job list, if all jobs in the job ticket are marked as finished in the job list
                    if all(list(validate_jobs(dict(name=alert['name'], status='finished')))):
                        clear_job_tickets(name=alert['name'])
                        # delete the worker
                        WorkerManager.delete_worker(alert['name'])
                    else:
                        logger.error(
                            f"Job {alert['job_id']} with job ticket {alert['name']} finished but the job list does not match. Please check the job logs.")
                    
                elif alert['status'] in ['FAILED', 'CANCELLED', 'TIMEOUT']:
                    logger.error(
                        f"Job {alert['job_id']} with job ticket {alert['name']} got status {str(alert['status'])}.")
                    # TODO: mark all not finished jobs in the job ticket as stopped
                    WorkerManager.delete_worker(alert['name'])

        if get_worker_number() >= WORKER_NUM:
            # if the number of workers is greater than or equal to the maximum number of workers, wait for a worker to finish
            continue

        # AFTER THIS POINT, A NEW WORKER CAN BE ADDED
        # get the number of workers to add
        num_add_workers = WORKER_NUM - get_worker_number()
        # make job tickets
        queue = make_priority_queue(job_list)
        # clear job tickets
        clear_job_tickets()
        # write job tickets
        ticket_dict = write_job_tickets(
            queue, num_jobs=PARALLEL_PROCS, num_tickets=num_add_workers)
        # add workers
        for i in range(num_add_workers):
            ticket = next(ticket_dict)
            name, jobs = ticket["id"], ticket["jobs"]
            # spawn a worker
            worker = WorkerManager.add_worker(name, jobs)
            # start the worker (this will submit the job to SLURM)
            worker.start()
            # await that the worker gets added to the queue, otherwise, log an error
            if not worker.await_queue():
                logger.error(
                    f"Worker {name} was not added to the queue. Timeout.")
                continue
            worker.update_status("running")

        # wait if the time taken is less than the scheduler interval, otherwise, continue
        end_time = time.time()
        time_taken = end_time - start_time
        if time_taken < SCHEDULER_INTERVAL:
            wait_time = SCHEDULER_INTERVAL - time_taken
            time.sleep(wait_time)
