#!/bin/bash      
#SBATCH --partition=main    
#SBATCH --job-name=elastic-db  
#SBATCH --nodes=1       
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=cpu-epyc-7
#SBATCH --mem=64g       
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/%j.out    
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err   

date;hostname;pwd   
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate   
cd /mnt/stud/work/kgutekunst/topic-analysis-text-data/ies-server
srun podman-compose down
srun podman-compose up