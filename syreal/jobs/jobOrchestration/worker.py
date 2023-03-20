import subprocess
import time
import pickle

from constants import TEMP_DIR, SLURM_TEMPLATE_FILE, SLURM_PARTITION, WORKER_TIMEOUT, PARALLEL_PROCS, EMAIL, SYREAL_PKG_DIR, PYTHON_SCRIPT_FILE, WORKER_MANAGER_PICKLE_FILE
from utils import Logger
from monitor import get_worker_names

logger, keep_fds = Logger()

class Worker:
    def __init__(self, name, jobs):
        self.jobs = jobs
        self.name = name
        self.job_id = None
        self.status = None

    def update_status(self, new_status, prev_status=None):
        from job_list_handler import write_job_list, update_jobs_status
        self.status = new_status
        if not prev_status:
            jobs_running = [dict(params=list(job.values())[0], status=new_status) for job in self.jobs]
            write_job_list(jobs_running, params_given=True)
        else:
            update_jobs_status(self.name, prev_status, new_status)
        


    def start(self):
        def write_worker_script(variables, filename):
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
            PARTITION=SLURM_PARTITION,
            TIMEOUT=WORKER_TIMEOUT,
            N_CPUS=PARALLEL_PROCS,
            EMAIL=EMAIL,
            SYREAL_PKG_DIR=SYREAL_PKG_DIR,
            PYTHON_SCRIPT_FILE=PYTHON_SCRIPT_FILE,
        ), filename=f"{self.name}.sh")

        # Submit shell script to SLURM
        cmd = ["sbatch", script_path]
        # TODO: delete script after submission
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if process.returncode != 0:
            logger.error(f"Failed to submit job to SLURM. Error: {err}")
        
        # Extract job ID from output
        self.job_id = out.decode().strip().split(" ")[-1]
        logger.info(f"Submitted job with ID {self.job_id} to SLURM.")

    def stop(self):
        # Cancel job
        cmd = ["scancel", self.job_id]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if process.returncode != 0:
            logger.error(f"Failed to submit job to SLURM. Error: {err}")
            return False
        else:
            self.update_status(new_status="stopped", prev_status="running")
            return True

    def await_queue(self):
        # Continuously check squeue for job status
        start_time = time.time()
        while True:
            cmd = ["squeue", "-j", self.job_id]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
            if process.returncode != 0:
                logger.error(f"Failed to get job status from SLURM. Error: {err}")
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
        with open(WORKER_MANAGER_PICKLE_FILE, "wb") as file:
            pickle.dump(self, file)

    def add_worker(self, name, jobs):
        worker = Worker(name, jobs)
        self.workers.append(worker)
        self.save()
        return worker

    def get_workers(self):
        return self.workers
    
    def get_worker(self, name):
        for worker in self.workers:
            if worker.name == name:
                return worker
        return None
    
    def validate(self):
        worker_names = set([worker.name for worker in self.workers])
        SLURM_worker_names = set(get_worker_names())
        # Check if any workers are missing
        mismatched_entries = list(SLURM_worker_names.symmetric_difference(worker_names))
        missing, extra = [mismatched_entries[:len(mismatched_entries)//2], mismatched_entries[len(mismatched_entries)//2:]]
        if len(mismatched_entries) > 0:
            logger.error(f"Worker names do not match. Running but not registered: {missing}, Registered but not running: {extra}")
            return False, missing, extra
        else:
            return True, [], []
        
    def delete_worker(self, name):
        for worker in self.workers:
            if worker.name == name:
                worker.stop()
                self.workers.remove(worker)
                break
    
    def delete_all_workers(self):
        for worker in self.workers:
            worker.stop()
        self.workers = []

    
def load_worker_manager():
    if not WORKER_MANAGER_PICKLE_FILE.exists():
        return WorkerManager()
    with open(WORKER_MANAGER_PICKLE_FILE, "rb") as file:
        WorkerManager = pickle.load(file)
    return WorkerManager

def delete_pickle_file():
    if WORKER_MANAGER_PICKLE_FILE.exists():
        WORKER_MANAGER_PICKLE_FILE.unlink()