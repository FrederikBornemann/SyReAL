import subprocess
import time
import pickle
import os

from constants import TEMP_DIR, SLURM_TEMPLATE_FILE, SLURM_PARTITION, WORKER_TIMEOUT, PARALLEL_PROCS, EMAIL, JOBS_DIR, PYTHON_SCRIPT_FILE, WORKER_MANAGER_PICKLE_FILE, WORKER_LOGS_DIR, ENV, SLURM_USER
from utils import Logger
from monitor import get_worker_names, get_worker_ids

logger, keep_fds = Logger()


class Worker:
    def __init__(self, name, jobs):
        self.jobs = jobs
        self.name = name
        self.job_id = None
        self.status = None

    def update_status(self, new_status):
        """Update the status of the worker."""
        self.status = new_status
        

    def start(self):
        """Start the worker by submitting a job to SLURM."""
        def write_worker_script(variables, filename):
            """Write a shell script to submit to SLURM."""
            import os
            # import the template
            with open(SLURM_TEMPLATE_FILE, 'r') as f:
                template = f.read()
            # Fill in the template with the variables
            script = template.format(**variables)
            # Create a temporary directory to save the script
            script_path = os.path.join(TEMP_DIR, filename)
            with open(script_path, 'w') as f:
                f.write(script)
            # Return the full path to the script
            return script_path
        script_path = write_worker_script(variables=dict(
            JOB_NAME=self.name,
            PARTITION=str(SLURM_PARTITION),
            TIMEOUT=str(WORKER_TIMEOUT),
            N_CPUS=str(PARALLEL_PROCS),
            EMAIL=str(EMAIL),
            DIR=str(JOBS_DIR),
            PYTHON_SCRIPT_FILE=str(PYTHON_SCRIPT_FILE),
            WORKER_LOGS_DIR=str(WORKER_LOGS_DIR),
            ENV=str(ENV),
            TIME=str(time.strftime("%H-%M-%S", time.localtime())),
        ), filename=f"{self.name}.sh")

        # Submit shell script to SLURM
        cmd = ["sbatch", script_path]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if process.returncode != 0:
            logger.error(
                f"Failed to submit job {self.name} to SLURM. Error: {err}")
            return False
        # Extract job ID from output
        self.job_id = out.decode().strip().split(" ")[-1]
        logger.info(
            f"Submitted job {self.name} with ID {self.job_id} to SLURM.")
        # delete script after submission
        os.remove(script_path)

    def stop(self):
        """Stop the worker by cancelling the job from SLURM."""
        # Update job list. This is done before because the worker can stop itself.
        # If the worker stops itself before updateing the job list, the job list will not be updated.
        # self.update_status(new_status="stopped", prev_status="running")
        # Cancel job
        if self.job_id is None:
            return True
        cmd = ["scancel", str(self.job_id)]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if process.returncode != 0:
            logger.error(
                f"Failed to cancel job with ID {self.job_id} ({self.name}) from SLURM . Error: {err}")
            return False
        else:
            return True

    def await_queue(self):
        """Wait for the job to be in the queue."""
        # Continuously check squeue for job status
        start_time = time.time()
        while True:
            cmd = ["squeue", "-j", str(self.job_id)]
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
            if process.returncode != 0:
                logger.error(
                    f"Failed to get job status from SLURM (job {self.name}). Error: {err}")
            # Check if job is in queue
            if len(out.decode().strip().split("\n")) > 1:
                break
            if time.time() - start_time > 30:
                return False
            time.sleep(5)
        return True


class WorkerManager:
    def __init__(self):
        self.workers = []

    # save worker manager as a pickle file
    def save(self):
        """Save the worker manager as a pickle file."""
        with open(WORKER_MANAGER_PICKLE_FILE, "wb") as file:
            pickle.dump(self, file)

    def add_worker(self, name, jobs):
        """Add a worker to the worker manager."""
        worker = Worker(name, jobs)
        self.workers.append(worker)
        self.save()
        return worker

    def get_workers(self):
        """Return a list of all workers."""
        return self.workers

    def get_worker_names(self):
        """Return a list of all worker names."""
        return [worker.name for worker in self.workers]

    def get_worker(self, name):
        """Return the worker with the given name."""
        for worker in self.workers:
            if worker.name == name:
                return worker
        return None

    def validate(self):
        """Validate the worker manager by checking if all workers are running and if all running workers are registered."""
        # TODO: validate workermanager and delete all workers that are not running
        worker_names = set([worker.name for worker in self.workers])
        SLURM_worker_names = set(get_worker_names(exclude_CG=True))
        # Check if any workers are missing
        mismatched_entries = list(
            SLURM_worker_names.symmetric_difference(worker_names))
        missing, extra = [mismatched_entries[:len(
            mismatched_entries)//2], mismatched_entries[len(mismatched_entries)//2:]]
        if len(mismatched_entries) > 0:
            logger.error(
                f"Worker names do not match. Running but not registered: {missing}, Registered but not running: {extra}")
            return False, missing, extra
        else:
            return True, [], []

    def delete_worker(self, name):
        """Delete the worker with the given name."""
        if not isinstance(name, list):
            name = [name]
        workers = self.workers.copy()
        for worker in workers:
            if worker.name in name:
                result = worker.stop()
                if result:
                    logger.debug(f"Stopped worker with name {worker.name}.")
                    self.workers.remove(worker)
                    self.save()
                yield result

    def delete_all_workers(self):
        """Delete all workers."""
        workers = [worker.name for worker in self.workers]
        logger.debug(f"Stopping workers with names {workers}.")
        stopping_success = list(self.delete_worker(workers))
        if not all(stopping_success):
            workers_not_stopped = [workers[i] for i, success in enumerate(
                stopping_success) if not success]
            logger.error(
                f"Failed to stop workers with names {workers_not_stopped}.")
            return False
        return True

    def kill_all_jobs(self):
        """Kill all jobs and update job list from "running" to "stopped"."""
        from job_list_handler import update_all_running_to_stopped
        # Cancel all jobs
        results = []
        job_ids = get_worker_ids(exclude_CG=True)
        for job_id in job_ids:
            cmd = ["scancel", str(job_id)]
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
            if process.returncode != 0:
                logger.error(
                    f"Failed to cancel job with ID {job_id} from SLURM. Error: {err}")
                results.append(False)
        if all(results):
            logger.debug("Successfully killed all jobs.")
            self.workers = []
            self.save()
            update_all_running_to_stopped()
            return True
        else:
            logger.error("Failed to kill all jobs.")
            self.workers = []
            self.save()
            return False

def stop_all_SLURM_nodes():
    # get all SLURM job ids
    job_ids = get_worker_ids(exclude_CG=True)
    # cancel all jobs
    for job_id in job_ids:
        cmd = ["scancel", str(job_id)]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if process.returncode != 0:
            logger.error(
                f"Failed to cancel job with ID {job_id} from SLURM. Error: {err}")

def load_worker_manager():
    """Load the worker manager from a pickle file."""
    if not WORKER_MANAGER_PICKLE_FILE.exists():
        return WorkerManager()
    with open(WORKER_MANAGER_PICKLE_FILE, "rb") as file:
        Worker_Manager = pickle.load(file)
    return Worker_Manager


def delete_pickle_file():
    """Delete the pickle file."""
    if WORKER_MANAGER_PICKLE_FILE.exists():
        WORKER_MANAGER_PICKLE_FILE.unlink()
