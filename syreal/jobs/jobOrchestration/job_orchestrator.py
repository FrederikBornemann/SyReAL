import time
import signal
import sys
import json

from priority_queue import job_tickets, clear_job_tickets, load_job_ticket
from utils import Logger
from monitor import get_worker_number
from worker import WorkerManager, Worker, load_worker_manager, delete_pickle_file, stop_all_SLURM_nodes
from job_monitor import write_job_dict, count_jobs

from constants import WORKER_NUM, PARALLEL_PROCS, SCHEDULER_INTERVAL, PROGRESS_FILE


def start_job_orchestrator():
    logger, keep_fds = Logger(add_handler=False)

    # Set the terminate flag to False
    global terminate
    terminate = False

    # Load WorkerManager from pickle file
    WorkerManager = load_worker_manager()

    def validate_worker_manager(WorkerManager):
        """Validate the workers by checking if the workers are running and if the workers are registered. If the workers are not running but registered, delete the workers. If the workers are running but not registered, add the workers."""
        valid, missing_workers, extra_workers = WorkerManager.validate()
        if not valid:
            logger.error(
                f"Worker names do not match. Running but not registered: {missing_workers}, Registered but not running: {extra_workers}")
            logger.warning("Deleting extra workers...")
            for worker_name in extra_workers:
                stopping_succeeded = WorkerManager.delete_worker(worker_name)
                # the variable stopping_succeeded should be False, because the worker should not be running
                if stopping_succeeded:
                    logger.debug(
                        f"Worker {worker_name} was running but has been deleted because it was monitored as running but not registered, this should not happen.")
            logger.warning(
                "Adding missing workers (if job ticket is still there)...")
            for worker_name in missing_workers:
                # TODO: change job tickets to current system
                job_ticket = load_job_ticket(worker_name)
                if job_ticket:
                    WorkerManager.add_worker(worker_name, job_ticket)
        return WorkerManager

    WorkerManager = validate_worker_manager(WorkerManager)

    def sigusr1_handler(signum, frame):
        logger.critical(
            "Received SIGUSR1 signal. Exiting the job orchestrator but not killing any jobs...")
        # Terminate the spawner, wait for the current job to finish, and then exit
        global terminate
        terminate = True
        # The terminate flag will be checked in the main loop, if it is True and the running jobs are finished, the job orchestrator will exit

    def sigterm_handler(signum, frame):
        logger.critical(
            "Received SIGTERM signal. Exiting the job orchestrator and killing all jobs...")
        # Killing all SLURM nodes, deleting the pickle file, and clearing the job tickets
        stop_all_SLURM_nodes()
        delete_pickle_file()
        clear_job_tickets()
        sys.exit(0)

    # Set the signal handler for SIGTERM and SIGKILL
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGUSR1, sigusr1_handler)

    logger.info("Starting the job orchestrator main loop...")
    # MAIN LOOP
    while True:
        # Get the current time for the scheduler
        start_time = time.time()

        # Compare the workers in the squeue with the workers in the WorkerManager. Fix any inbalances.
        WorkerManager = validate_worker_manager(WorkerManager)

        # make updated job status dict
        status_dict = write_job_dict()
        # get the number of jobs in each status
        status_counter = count_jobs(status_dict)

        # if all jobs are finished, exit the job orchestrator
        num_all_jobs = sum(status_counter.values())
        num_finished_jobs = status_counter["finished"]
        if num_finished_jobs == num_all_jobs:
            logger.info(
                "DONE!! All jobs finished. Exiting the job orchestrator...")
            break

        # if prev_status_counter is not defined, define it
        if 'prev_status_counter' not in locals():
            prev_status_counter = dict()
        # save the status counter if the counter has changed
        if status_counter != prev_status_counter:
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(status_counter, f)
        prev_status_counter = status_counter

        # check if there are free workers
        num_running_workers = get_worker_number()

        # check if there are free workers
        num_running_workers = get_worker_number()
        num_running_jobs = status_counter["running"]
        if (num_running_workers >= WORKER_NUM) or terminate or (num_all_jobs-num_finished_jobs-num_running_jobs == 0):
            # if the number of workers is greater than or equal to the maximum number of workers, wait for a worker to finish
            if (num_running_workers == 0) and terminate:
                # if there are no workers running and the terminate flag is set, exit the job orchestrator
                logger.info(
                    "All jobs have finished. Exiting the job orchestrator.")
                sys.exit(0)
        else:
            # AFTER THIS POINT, A NEW WORKER CAN BE ADDED
            # get the number of workers to add
            num_add_workers = WORKER_NUM - get_worker_number()

            logger.info(
                f"Adding {num_add_workers} worker{'s' if num_add_workers>1 else ''}...")

            # calculate the number of jobs that are not running (remaining jobs)
            not_running_jobs_count = status_counter["pending"] + \
                status_counter["stopped"]

            # if there are no jobs left, wait for a worker to finish
            if not_running_jobs_count == 0:
                logger.info(
                    "No jobs left to add. Waiting for a worker to finish...")
                continue
            # if the potential number of jobs to add is greater than the number of not running jobs, reduce the number of workers to add
            if num_add_workers * PARALLEL_PROCS > not_running_jobs_count:
                logger.info(
                    f"Could not add {num_add_workers} workers with {PARALLEL_PROCS} jobs each. Adding {-(-not_running_jobs_count // PARALLEL_PROCS)} workers with {PARALLEL_PROCS} jobs each. Number of jobs to distribute: {not_running_jobs_count}.")
                # round up using integer division and negation
                num_add_workers = -(-not_running_jobs_count // PARALLEL_PROCS)
            # add workers
            for i, ticket in enumerate(job_tickets(status_dict=status_dict, ticket_num=num_add_workers)):
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
                WorkerManager.save()
                if terminate:
                    break

        # wait if the time taken is less than the scheduler interval, otherwise, continue
        time_taken = time.time() - start_time
        if time_taken < SCHEDULER_INTERVAL:
            time.sleep(SCHEDULER_INTERVAL - time_taken)
