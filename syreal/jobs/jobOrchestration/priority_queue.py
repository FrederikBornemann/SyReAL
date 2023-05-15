from utils import generate_id, Logger
from constants import JOB_DICT, PARALLEL_PROCS, JOB_TICKETS, BATCH_DIR, BATCH_SIZE, BATCH_COUNT, WORKER_NUM
from utils import timeit_wrapper
import pickle
logger, keep_fds = Logger(add_handler=False)


@timeit_wrapper
def make_priority_queue(status_dict=None) -> list[tuple[tuple[str, str, str], int]]:
    """Returns a sorted priority queue of jobs. The priority queue is a list of tuples ((equation, algorithm, trial), priority)."""
    # load the job dict
    if not status_dict:
        # check if the job dict exists
        if not JOB_DICT.exists():
            logger.error(
                f"func: make_priority_queue :   Job dict {JOB_DICT} does not exist.")
            return []
        with open(JOB_DICT, "rb") as f:
            status_dict = pickle.load(f)
    # make the priority queue
    priority_queue = []
    for key in status_dict:
        # skip finished and running jobs
        if status_dict[key][0] in ["pending", "stopped"]:
            # stopped jobs have priority 0, pending jobs have priority 1
            prio = 0 if status_dict[key][0] == "stopped" else 1
            priority_queue.append((key, prio))
    # sort the priority queue by priority
    priority_queue.sort(key=lambda x: x[1])
    return priority_queue


def job_tickets(status_dict=None, jobs=None, ticket_num=None, jobs_in_ticket_num=PARALLEL_PROCS, save_to_file=True, refill_batches=True, ticket_directory=JOB_TICKETS) -> dict[str, list]:
    """Yields a list of jobs for each ticket. The number of tickets is determined by the ticket_num parameter. The number of jobs in each ticket is determined by the jobs_in_ticket_num parameter. If the save_to_file parameter is True, the job tickets are saved to files in JOB_TICKETS."""
    if not jobs:
        jobs = make_priority_queue(status_dict=status_dict)
    num_slices = (len(jobs) + jobs_in_ticket_num - 1) // jobs_in_ticket_num
    num_slices = min(
        num_slices, ticket_num) if ticket_num is not None else num_slices
    # refill the batch directory if necessary
    if refill_batches and len(jobs[num_slices*jobs_in_ticket_num:]) > 0:
        refill_batch_jobs(
            jobs=jobs[num_slices*jobs_in_ticket_num:])
    # generate the job tickets
    for i, _ in enumerate(range(num_slices)):
        jobs_slice = jobs[i*jobs_in_ticket_num:(i+1)*jobs_in_ticket_num]
        # generate an ID for the ticket
        ID = generate_id(jobs_slice)
        # save the ticket to a file in JOB_TICKETS
        job_ticket = dict(id=ID, jobs=jobs_slice)
        if save_to_file:
            with open(JOB_TICKETS / f"{ID}.pickle", "wb") as f:
                pickle.dump(job_ticket, f)
        yield job_ticket


def refill_batch_jobs(jobs, update_batches=True):
    """Refills the batch directory with job tickets."""
    # if batch directory is full, do nothing. If update_batches is True, update the batches.
    if (not update_batches) and len(list(BATCH_DIR.iterdir())) >= WORKER_NUM * BATCH_COUNT:
        return
    # delete all files in the batch directory
    for file in BATCH_DIR.iterdir():
        file.unlink()
    # save the job tickets to the batch directory
    for i, ticket in enumerate(job_tickets(jobs=jobs, ticket_num=WORKER_NUM * BATCH_COUNT, jobs_in_ticket_num=BATCH_SIZE, save_to_file=True, refill_batches=False, ticket_directory=BATCH_DIR)):
        pass


def clear_job_tickets(ticket_name=None):
    """Clears the job tickets. If the ticket_name parameter is not None, only the ticket with the given name is cleared."""
    if ticket_name is None:
        for file in JOB_TICKETS.iterdir():
            file.unlink()
        for file in BATCH_DIR.iterdir():
            file.unlink()
    else:
        (JOB_TICKETS / f"{ticket_name}.pickle").unlink()


def load_job_ticket(ticket_name):
    """Loads the job tickets with the given name."""
    if not (JOB_TICKETS / f"{ticket_name}.pickle").exists():
        return None
    with open(JOB_TICKETS / f"{ticket_name}.pickle", "rb") as f:
        job_ticket = pickle.load(f)
    return job_ticket


if __name__ == "__main__":
    priority_queue = make_priority_queue()
    print(priority_queue)
    print(len(priority_queue))
    for i, ticket in enumerate(job_tickets(ticket_num=2)):
        print(i, ticket, len(ticket))
