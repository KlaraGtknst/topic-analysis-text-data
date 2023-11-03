#!/bin/bash      
#SBATCH --partition=main        
#SBATCH --job-name=flask 
#SBATCH --nodes=1    
#SBATCH --cpus-per-task=1 
#SBATCH --nodelist=cpu-epyc-7
#SBATCH --mem=8g    
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/%j.out      
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err   
# (%N: Nodename, %j: Job-Nr.)

date;hostname;pwd   
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate  
cd /mnt/stud/work/kgutekunst/topic-analysis-text-data/
python -m flask run --app server --host=0.0.0.0 --port 8000