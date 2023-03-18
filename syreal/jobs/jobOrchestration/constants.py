from pathlib import Path
import os

PKG_DIR = Path(__file__).parents[3]
JOBS_DIR = Path(os.environ.get('JOB_ORCHESTRATION_WORKSPACE'))
TEMP_DIR = JOBS_DIR / "tmp"
JOB_LIST = JOBS_DIR / "job_list.json"
JOB_TICKETS = JOBS_DIR / "job_tickets"
SLURM_USER = os.environ.get('SLURM_USER')
SLURM_PARTITION = os.environ.get('SLURM_PARTITION')
PID_DIR = TEMP_DIR / "pid"
PID_FILE = PID_DIR / "job_scheduler.pid"