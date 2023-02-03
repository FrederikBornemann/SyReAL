#!/bin/bash

#SBATCH -e logs/%J.err
#SBATCH -o logs/%J.out
#SBATCH -J syreal_example

#SBATCH --mail-type=END
#SBATCH --mail-user=frederik.johannes.bornemann@desy.de
#SBATCH --partition=allcpu
#SBATCH --time=2:00:00
#SBATCH --nodes=1



###
source ~/.bashrc
conda activate PySR
cd /home/bornemaf/SyReAL
python example.py
