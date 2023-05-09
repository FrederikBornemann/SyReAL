import time
import signal
import sys

from priority_queue import make_priority_queue
from utils import Logger
from monitor import get_worker_number
from worker import WorkerManager, Worker, load_worker_manager, delete_pickle_file
from email_scraper import check_for_new_events
from job_list_handler import validate_jobs, read_job_list, clear_job_tickets, write_job_tickets, update_jobs_status, generate_data_dirs, get_job_ticket, backup_job_list, replace_job_list_file
from progress import get_progress

from constants import WORKER_NUM, PARALLEL_PROCS, SCHEDULER_INTERVAL, BACKUP_INTERVAL


def start_job_orchestrator():
    logger, keep_fds = Logger(add_handler=False)

    # Set the terminate flag to False
    global terminate
    terminate = False

    # Load WorkerManager from pickle file
    WorkerManager = load_worker_manager()
    # Validate the workers by checking if the workers are running and if the workers are registered
    # If the workers are not running but registered, delete the workers
    # If the workers are running but not registered, add the workers
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
        logger.warning("Adding missing workers (if job ticket is still there)...")
        for worker_name in missing_workers:
            job_ticket = get_job_ticket(worker_name)
            if job_ticket:
                WorkerManager.add_worker(worker_name, job_ticket)

    def sigusr1_handler(signum, frame):
        logger.critical(
            "Received SIGUSR1 signal. Exiting the job orchestrator but not killing any jobs...")
        # Terminate the spawner, wait for the current job to finish, and then exit
        global terminate
        terminate = True
        # The terminate flag will be checked in the main loop, if it is True and the running jobs are finished, the job orchestrator will exit

    def sigterm_handler(signum, frame):
        logger.critical(
            "Received SIGTERM signal. Killing all jobs, deleting all job tickets and workers, and marking all running jobs as stopped...")
        # Kill all jobs, delete all job tickets, and mark all running jobs as stopped, delete WorkerManager pickle file
        # TODO: delete all workers => should be done
        all_stopped = False
        while not all_stopped:
            all_stopped = WorkerManager.delete_all_workers()
        # validate that all workers have been deleted
        assert len(WorkerManager.get_workers()) == 0
        WorkerManager.kill_all_jobs()
        delete_pickle_file()
        clear_job_tickets(all=True)
        sys.exit(0)

    # Set the signal handler for SIGTERM and SIGKILL
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGUSR1, sigusr1_handler)

    # Start the job orchestrator
    logger.info("Job orchestrator has started.")

    # Generate the data directories
    logger.info("Generating data directories...")
    generate_data_dirs()
    logger.info("Checking validity of job list...")
    replace_job_list_file()

    logger.info("Starting the job orchestrator main loop...")
    counter = 0
    inbalanced_counter = 0
    
    # MAIN LOOP
    while True:
        # If the while loop counter reaches a certain number, then backup the job list
        if counter >= BACKUP_INTERVAL:
            backup_job_list()
            counter = 0
        counter += 1

        # check if the terminate flag is set
        start_time = time.time()
        # check if there are any new events
        try:
            alerts = check_for_new_events()
        except Exception as e:
            logger.error(f"Error while checking emails for new events: {e}")
            alerts = []
        if alerts:
            for alert in alerts:
                # check if the job name is in worker names
                if not alert['name'] in WorkerManager.get_worker_names():
                    logger.warning(
                        f"Job {alert['job_id']} with job ticket {alert['name']} is not registered in the job orchestrator but was alerted by email. Please check the job logs. This can be ignored if the job was cancelled in a previous run.")
                    continue
                if alert['status'] == 'COMPLETED':
                    # remove the job ticket from the job list, if all jobs in the job ticket are marked as finished in the job list
                    if all(list(validate_jobs(dict(name=alert['name'], status='finished')))):
                        logger.info(
                            f"Job {alert['job_id']} with job ticket {alert['name']} finished. Deleting job ticket and worker.")
                        # delete the worker
                        WorkerManager.delete_worker(alert['name'])
                        clear_job_tickets(name=alert['name'])
                    else:
                        logger.error(
                            f"Job {alert['job_id']} with job ticket {alert['name']} finished but the job list does not match. Please check the job logs.")
                    
                elif alert['status'] in ['FAILED', 'CANCELLED', 'TIMEOUT']:
                    logger.error(
                        f"Job {alert['job_id']} with job ticket {alert['name']} got status {str(alert['status'])}. Marking all running jobs from {alert['name']} as 'stopped'.")
                    # mark all running jobs from the stopped worker as 'stopped'
                    update_jobs_status(alert['name'], prev_status='running', new_status='stopped')
                    WorkerManager.delete_worker(alert['name'])
                    clear_job_tickets(name=alert['name'])
            WorkerManager.save()

        
        # check if there are free workers
        num_running_workers = get_worker_number()
        # get progress
        while True:
            try:
                num_finished_jobs, num_running_jobs, num_all_jobs = get_progress()
                break
            except ValueError as e:
                logger.error(f"Error while getting progress: {e}")
                # replace the job list file with the backup
                replace_job_list_file()
        if num_finished_jobs == num_all_jobs:
            logger.info("DONE!! All jobs finished. Exiting the job orchestrator...")
            break
        if num_running_jobs > num_running_workers * PARALLEL_PROCS:
            inbalanced_counter += 1
            if inbalanced_counter >= 5:
                inbalanced_counter = 0
                logger.error(
                    f"Number of running jobs ({num_running_jobs}) is greater than the number of workers ({num_running_workers}) * PARALLEL_PROCS ({PARALLEL_PROCS}). This should not happen.")
                logger.error("Killing all jobs, deleting all job tickets and workers, and marking all running jobs as stopped...")
                if WorkerManager.kill_all_jobs():
                    clear_job_tickets(all=True)
                else:
                    logger.error("Could not kill all jobs...")
            else:
                logger.warning(
                    f"Number of running jobs ({num_running_jobs}) is greater than the number of workers ({num_running_workers}) * PARALLEL_PROCS ({PARALLEL_PROCS}). This should not happen.")
        else:
            inbalanced_counter = 0
        # check if there are free workers
        num_running_workers = get_worker_number()
        if (num_running_workers >= WORKER_NUM) or terminate or (num_all_jobs-num_finished_jobs-num_running_jobs == 0):
            # if the number of workers is greater than or equal to the maximum number of workers, wait for a worker to finish
            if (num_running_workers == 0) and terminate:
                # if there are no workers running and the terminate flag is set, exit the job orchestrator
                logger.info("All jobs have finished. Exiting the job orchestrator.")
                sys.exit(0)
        else:
            # AFTER THIS POINT, A NEW WORKER CAN BE ADDED
            # get the number of workers to add
            num_add_workers = WORKER_NUM - get_worker_number()
            logger.info(f"Adding {num_add_workers} worker{'s' if num_add_workers>1 else ''}...")
            # read the job list
            job_list = read_job_list()
            # make job tickets
            # get the number of jobs to add. If there are not enough jobs left, add all remaining jobs
            num_jobs = PARALLEL_PROCS * num_add_workers if num_all_jobs-num_finished_jobs-num_running_jobs > PARALLEL_PROCS * num_add_workers else num_all_jobs-num_finished_jobs-num_running_jobs
            if num_jobs == 0:
                logger.info("No jobs left to add. Waiting for a worker to finish...")
                continue
            # If there are not enough jobs left, add all remaining jobs. This is not a optimal solution, but it should work for now.
            while True:
                try:
                    queue = make_priority_queue(job_list, num_jobs)
                except IndexError:
                    logger.error(
                        f"Could not make a priority queue with {num_jobs} jobs. Trying with {num_jobs-1} jobs...")
                    num_jobs -= 1
                    continue
                else:
                    break
                
            if num_jobs < PARALLEL_PROCS * num_add_workers:
                logger.warning(
                    f"Could not add {num_add_workers} workers with {PARALLEL_PROCS} jobs each. Adding {-(-num_jobs // PARALLEL_PROCS)} workers with {PARALLEL_PROCS} jobs each. Number of jobs to distribute: {num_jobs}.")
                num_add_workers = -(-num_jobs // PARALLEL_PROCS)
            # write job tickets
            ticket_dict = write_job_tickets(
                queue, num_jobs=-(-num_jobs // num_add_workers), num_tickets=num_add_workers)
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
                WorkerManager.save()
                if terminate:
                    break

        # wait if the time taken is less than the scheduler interval, otherwise, continue
        time_taken = time.time() - start_time
        if time_taken < SCHEDULER_INTERVAL:
            time.sleep(SCHEDULER_INTERVAL - time_taken)
