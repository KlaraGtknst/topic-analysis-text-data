#!/bin/bash      
#SBATCH --partition=main        # Partition main
#SBATCH --job-name=init-db  # Job-Name
#SBATCH --nodes=1       # 1 Node wird benötigt
#SBATCH --cpus-per-task=45 
#SBATCH --nodelist=cpu-epyc-7
#SBATCH --mem=264g       # 264 GB Hauptspeicher
#SBATCH --time=100:00:00  # max. Laufzeit 100h
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/%j.out        # Datei für stdout (logs/ prints != results, e.g., .pdf files) 
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err     # Datei für stderr
# (%N: Nodename, %j: Job-Nr.)

date;hostname;pwd    # Ausgabe des Datums, des Hostnamens und des Arbeitsverzeichnisses
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate    # virtuelle Umgebung aktivieren

srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'create_database.py' -p 1 -a 'http://cpu-epyc-7.ies.uni-kassel.de:9200' -d '/mnt/datasets/Bahamas/SAC/0/*.pdf' -D '/mnt/stud/home/kgutekunst/visualizations/images/'
srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'create_documents.py' -p 1 -a 'http://cpu-epyc-7.ies.uni-kassel.de:9200' -d '/mnt/datasets/Bahamas/SAC/0/*.pdf' -D '/mnt/stud/home/kgutekunst/visualizations/images/'

srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'insert_embeddings.py' -m 'infer' -p 1 -a 'http://cpu-epyc-7.ies.uni-kassel.de:9200' -d '/mnt/datasets/Bahamas/SAC/0/*.pdf' -D '/mnt/stud/home/kgutekunst/visualizations/images/'
srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'insert_embeddings.py' -m 'doc2vec' -p 1 -a 'http://cpu-epyc-7.ies.uni-kassel.de:9200' -d '/mnt/datasets/Bahamas/SAC/0/*.pdf' -D '/mnt/stud/home/kgutekunst/visualizations/images/'
srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'insert_embeddings.py' -m 'hugging' -p 1 -a 'http://cpu-epyc-7.ies.uni-kassel.de:9200' -d '/mnt/datasets/Bahamas/SAC/0/*.pdf' -D '/mnt/stud/home/kgutekunst/visualizations/images/'
srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'insert_embeddings.py' -m 'tfidf' -p 1 -a 'http://cpu-epyc-7.ies.uni-kassel.de:9200' -d '/mnt/datasets/Bahamas/SAC/0/*.pdf' -D '/mnt/stud/home/kgutekunst/visualizations/images/'
srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'insert_embeddings.py' -m 'universal' -p 1 -a 'http://cpu-epyc-7.ies.uni-kassel.de:9200' -d '/mnt/datasets/Bahamas/SAC/0/*.pdf' -D '/mnt/stud/home/kgutekunst/visualizations/images/'
srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'insert_clusters.py' -m 'none' -p 1 -a 'http://cpu-epyc-7.ies.uni-kassel.de:9200' -d '/mnt/datasets/Bahamas/SAC/0/*.pdf' -D '/mnt/stud/home/kgutekunst/visualizations/images/'