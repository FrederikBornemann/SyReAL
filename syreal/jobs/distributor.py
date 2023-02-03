# distributor.py

# Jobs:
# - submit number of jobs to get the number of running procs (nrp) to desired value (gets called by watcher.py when nrp < desired)

import json
import os
from job_list import make_job_list
import random
import subprocess
import re



def template(kwargs):
    return f"""\
#!/bin/bash

#########################
## SLURM JOB COMMANDS ###
#########################
#SBATCH --partition=allcpu           ## or allgpu / cms / cms-uhh / maxgpu
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --job-name {kwargs["job_name"]}           # give job unique name
#SBATCH --output ./logs/training-%j.out      # terminal output
#SBATCH --error ./logs/training-%j.err
#SBATCH --no-requeue
#SBATCH --mincpus={kwargs["mincpus"]}
##SBATCH --mail-type END
##SBATCH --mail-user frederik.johannes.bornemann@desy.de



#####################
### BASH COMMANDS ###
#####################

## examples:

# source .bashrc for init of anaconda
source ~/.bashrc

# activate your conda environment the job should use
conda activate PySR

# go to your folder with your python scripts
cd {os.getcwd()}

# run
{kwargs["scripts"]}
    """
def make_python_call(seeds, algos, infos, parentdir, dirs, boundaries, equation):
    boundaries = "'{"+str(boundaries).replace(" ", "").replace("'", r'\"')+"}'"
    # Using regex: convert the equation string so it can be interpreted by argparse
    equation = re.sub(r"(?<=[()+*/^-])(\d+)(?=[()+*/^-])", lambda x: re.sub(r'(\d+)',r'\1.0', x.group(1)), str(equation)).replace("(", r'\(').replace(")", r'\)')
    lst = list()
    for seed, algo, info in zip(seeds, algos, infos):
        lst.append(f'python listener.py --seed {seed} --algorithm {algo} --info {info} --parentdir {dirs["data"]} --monitor {dirs["monitor"]}  --equation {equation} &') #--boundaries {boundaries}
    seeds = [str(x) for x in seeds]
    lst.append(f"python node_watcher.py --seeds {' '.join(seeds)} --algorithms {' '.join(algos)} --monitor {dirs['monitor']}")
    return "\n".join(lst)

def exec_slurm_scripts(jobs, parentdir, dirs, boundaries, equation):
    if len(jobs) <= 0:
        print("ERROR: len(jobs) < 0. No jobs submitted")
        return jobs
    for i, job in enumerate(jobs):
        seeds, algos, infos = job["seed"], job["algorithm"], job["info"]
        scripts = make_python_call(seeds, algos, infos, parentdir, dirs, boundaries, equation)
        with open (f'run_script_{i}.sh', 'w') as rsh:
            rsh.write(template(dict(job_name=f"PySR_SyReAL_trails_{'%05d' % random.randint(0,99999)}", mincpus=60, scripts=scripts)))
        os.system(f'sbatch run_script_{i}.sh')
        if os.path.exists(f'run_script_{i}.sh'):
            os.remove(f'run_script_{i}.sh')
            pass
    return
        

def Distributor(jobs, parentdir, dirs, boundaries, equation): 
    # make and execute slurm scripts
    exec_slurm_scripts(jobs, parentdir, dirs, boundaries, equation)
    return

