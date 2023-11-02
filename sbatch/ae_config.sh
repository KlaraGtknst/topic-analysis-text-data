#!/bin/bash      
#SBATCH --partition=main    
#SBATCH --job-name=par-ae-config
#SBATCH --nodes=1    
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=cpu-epyc-5
#SBATCH --mem=66g 
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/%j.out   
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err    
# (%N: Nodename, %j: Job-Nr.)

date;hostname;pwd  
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate  
srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'opt_architecture_ae.py' -p 8 -d '/mnt/datasets/Bahamas/*.pdf' -D '/mnt/stud/home/kgutekunst/visualizations/images/'