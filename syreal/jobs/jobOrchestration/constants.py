from pathlib import Path
import yaml

SYREAL_PKG_DIR = Path(__file__).parents[3]
PKG_DIR = Path(__file__).parents[0]
CONFIG_FILE = PKG_DIR / "config.yaml"

# CONFIG
try:
    with open(CONFIG_FILE, 'r') as yml:
        config = yaml.safe_load(yml)
except Exception:
    raise Exception("Config file not found. Please create a config.yaml file in the jobOrchestration directory.")

for key, value in config.items():
    globals()[key] = value

JOBS_DIR = Path(config['JOBS_DIR'])
TEMP_DIR = JOBS_DIR / "tmp"
JOB_LIST = JOBS_DIR / "job_list.json"
JOB_TICKETS = JOBS_DIR / "job_tickets"
PID_DIR = TEMP_DIR / "pid"
PID_FILE = PID_DIR / "job_scheduler.pid"
PID_LOGS_DIR = TEMP_DIR / "logs"
PID_LOG_FILE = PID_LOGS_DIR / "job_scheduler.log"
SLURM_TEMPLATE_FILE = PKG_DIR / "slurm_template.sh"
WORKER_MANAGER_PICKLE_FILE = TEMP_DIR / "worker_manager.pickle"
FEYNMAN_CSV_FILE = PKG_DIR / "feynman.csv"
WORKER_OUTPUT_DIR = JOBS_DIR / "data"


