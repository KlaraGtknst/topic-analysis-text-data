#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/jupyter.log
#SBATCH --ntasks=1
#SBATCH --mem=60gb
#SBATCH --cpus-per-task=1
#SBATCH --partition=jupyter
#SBATCH --gres=gpu:1
date;hostname;pwd
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate
cd /mnt/stud/work/kgutekunst/topic-analysis-text-data
srun jupyter lab --port=9711 --no-browser