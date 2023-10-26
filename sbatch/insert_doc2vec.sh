#!/bin/bash      
#SBATCH --partition=main   
#SBATCH --job-name=doc2vec  
#SBATCH --nodes=1      
#SBATCH --cpus-per-task=128
#SBATCH --nodelist=cpu-epyc-7
#SBATCH --mem=200g
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/%j.out        
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err    
# (%N: Nodename, %j: Job-Nr.)

date;hostname;pwd 
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate    

srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'insert_embeddings.py' -m 'doc2vec' -p 128 -a 'http://cpu-epyc-7.ies.uni-kassel.de:9200' -d '/mnt/datasets/Bahamas/*.pdf' -D '/mnt/stud/home/kgutekunst/visualizations/images/'