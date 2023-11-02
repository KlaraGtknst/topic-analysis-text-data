#!/bin/bash      
#SBATCH --partition=main        # Partition main
#SBATCH --job-name=sbert-hf  # Job-Name
#SBATCH --nodes=1      
#SBATCH --cpus-per-task=36
#SBATCH --nodelist=cpu-epyc-7
#SBATCH --mem=200g
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/%j.out        
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err    
# (%N: Nodename, %j: Job-Nr.)

date;hostname;pwd 
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate  

srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'insert_embeddings.py' -m 'hugging' -p 36 -a 'http://cpu-epyc-7.ies.uni-kassel.de:9200' -d '/mnt/datasets/Bahamas/*.pdf' -D '/mnt/stud/home/kgutekunst/visualizations/images/'