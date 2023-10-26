#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/jupyter.log
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=10
#SBATCH --partition=jupyter
#SBATCH --gres=gpu:1
date;hostname;pwd
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate
cd /mnt/stud/home/kgutekunst
srun jupyter lab --port=9711 --no-browser