#!/bin/bash      
#SBATCH --partition=main      
#SBATCH --job-name=tfidf-ae 
#SBATCH --nodes=1    
#SBATCH --cpus-per-task=45 
#SBATCH --nodelist=cpu-epyc-6
#SBATCH --mem=264g 
#SBATCH --time=100:00:00  
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/%j.out    
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err     

date;hostname;pwd    
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate  
srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'insert_embeddings.py' -m 'tfidf' -p 1 -a 'http://cpu-epyc-6.ies.uni-kassel.de:9200' -d '/mnt/datasets/Bahamas/SAC/0/*.pdf' -D '/mnt/stud/home/kgutekunst/visualizations/images/'