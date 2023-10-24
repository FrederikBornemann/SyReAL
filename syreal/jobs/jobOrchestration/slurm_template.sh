#!/bin/bash

#########################
## SLURM JOB COMMANDS ###
#########################
#SBATCH --partition={PARTITION}
#SBATCH --time={TIMEOUT}
#SBATCH --nodes=1
#SBATCH --job-name {JOB_NAME}
#SBATCH --output {WORKER_LOGS_DIR}/{JOB_NAME}_{TIME}.out
#SBATCH --error {WORKER_LOGS_DIR}/{JOB_NAME}_{TIME}.err
#SBATCH --no-requeue
#SBATCH --mincpus={N_CPUS}
#SBATCH --mail-type END
#SBATCH --mail-user {EMAIL}

source ~/.bashrc
conda activate {ENV}
cd {DIR}
python {PYTHON_SCRIPT_FILE} --worker_name={JOB_NAME}
