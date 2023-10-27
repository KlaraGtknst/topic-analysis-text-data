#!/bin/bash      
#SBATCH --partition=main    
#SBATCH --job-name=pca-config
#SBATCH --nodes=1    
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=cpu-epyc-1
#SBATCH --mem=266g 
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/%j.out   
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err    
# (%N: Nodename, %j: Job-Nr.)

date;hostname;pwd  
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate  
srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'num_pca_comp.py' -p 1 -d '/mnt/datasets/Bahamas/*.pdf' -D '/mnt/stud/home/kgutekunst/visualizations/images/'