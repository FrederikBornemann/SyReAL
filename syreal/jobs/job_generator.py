# generates a job directory with a monitor.json, a config.yml file and a outpuut folder
# It takes a custom job_{name}.yml file as input. There is a template file in the directory.

import yaml
from pathlib import Path
import os
import json
from read_config import read_config

# read package config.yml
pkg_config = read_config()

PKG_DIR = Path(__file__).parents[2]
JOBS_DIR = PKG_DIR / "job_templates"
JOB_TEMPLATE = JOBS_DIR / "job_template.yml"
OUTPUT_DIR = Path(config["directory"]["outputParentDirectory"])


def load_job_yaml(path=JOBS_TEMPLATE):
    with open(path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def generate_jobs(config):
    job_dir = OUTPUT_DIR / ""
    job_dir.mkdir(parents=True, exist_ok=True)


