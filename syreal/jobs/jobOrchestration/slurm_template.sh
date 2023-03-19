#!/bin/bash

#########################
## SLURM JOB COMMANDS ###
#########################
#SBATCH --partition=$PARTITION
#SBATCH --time=$TIMEOUT
#SBATCH --nodes=1
#SBATCH --job-name $JOB_NAME
#SBATCH --output ./logs/training-%j.out
#SBATCH --error ./logs/training-%j.err
#SBATCH --no-requeue
#SBATCH --mincpus=$N_CPUS
##SBATCH --mail-type END
##SBATCH --mail-user $EMAIL

source ~/.bashrc
conda activate PySR
cd $SYREAL_PKG_DIR
python $PYTHON_SCRIPT_FILE
