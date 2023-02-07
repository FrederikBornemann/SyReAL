# generates a job directory with a monitor.json, a config.yml file and a outpuut folder
# It takes a custom job_{name}.yml file as input. There is a template file in the directory.

import yaml
from pathlib import Path
import json
from read_config import read_config
import shutil

PKG_DIR = Path(__file__).parents[2]
JOBS_DIR = PKG_DIR / "job_templates"
JOB_TEMPLATE = JOBS_DIR / "job_template.yml"

pkg_config = read_config()


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    import sys
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
            
def feynman_job(config):

def master_job(config):
    master_general = config["GENERAL"]
    JOB_OUTPUT_DIR = Path(pkg_config["directory"]["outputParentDirectory"]) / master_general["Output"]
    try:
        (JOB_OUTPUT_DIR / "data").mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if query_yes_no(f"Job {master_general['JobName']} already exists. Do you want to overwrite?"):
            shutil.rmtree(JOB_OUTPUT_DIR)
            (JOB_OUTPUT_DIR / "data").mkdir(parents=True, exist_ok=True)
        else:
            raise Exception("Job already exists. Create new job yaml or change the JobName of the current one.")

def worker_job(config):
    # make directory
    JOB_OUTPUT_DIR = Path(pkg_config["directory"]["outputParentDirectory"]) / config["GENERAL"]["Output"]
    try:
        (JOB_OUTPUT_DIR / "data").mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if query_yes_no(f"Job {config['GENERAL']['JobName']} already exists. Do you want to overwrite?"):
            shutil.rmtree(JOB_OUTPUT_DIR)
            (JOB_OUTPUT_DIR / "data").mkdir(parents=True, exist_ok=True)
        else:
            raise Exception("Job already exists. Create new job yaml or change the JobName of the current one.")
    # make jobs.json
    def in_range(n, range_string):
        try:
            start, end = map(int, str(range_string).split("-"))
        except ValueError:
            start = end = int(range_string)
        return start <= n <= end
    def info(i):
        kwargs, slurm = config["GENERAL"]["kwargs"].copy(), config["GENERAL"]["slurm_kwargs"].copy()
        print(i, kwargs, slurm)
        i_in_sections = list(map(lambda range: in_range(i, range), l))
        if i_in_sections.count(True) == 1:
            section = list(config["JOBS"].keys())[i_in_sections.index(True)]
        else:
            raise Exception("Job number not in range of any section")
        kwargs.update(config["JOBS"][section]["kwargs"])
        slurm.update(config["JOBS"][section]["slurm_kwargs"])
        return dict(role="WORKER", status="pending", kwargs=kwargs, slurm_kwargs=slurm)

    jobs_dict = dict(jobs={i: info(i) for i in range(1, config["GENERAL"]["JobNumber"] + 1)}, progress=f"0/{config['GENERAL']['JobNumber']}", running = f'0/{config["GENERAL"]["NodeNumber"]}')
    return jobs_dict, JOB_OUTPUT_DIR

def generate_jobs(path=JOB_TEMPLATE):
    config = read_config(path)
    role = config["GENERAL"]["Role"]
    if role == "MASTER":
        return master_job(config)
    elif role == "WORKER":
        return worker_job(config)
    
    with open(JOB_OUTPUT_DIR / "jobs.json", "w") as file:
        json.dump(jobs_dict, file)
    return JOB_OUTPUT_DIR
