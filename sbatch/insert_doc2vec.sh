#!/bin/bash      
#SBATCH --partition=main        # Partition main
#SBATCH --job-name=doc2vec  # Job-Name
#SBATCH --nodes=1       # 1 Node wird benötigt
#SBATCH --cpus-per-task=45 
#SBATCH --nodelist=cpu-epyc-1
#SBATCH --mem=264g       # 264 GB Hauptspeicher
#SBATCH --time=100:00:00  # max. Laufzeit 100h
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/%j.out        # Datei für stdout (logs/ prints != results, e.g., .pdf files) 
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err     # Datei für stderr
# (%N: Nodename, %j: Job-Nr.)

date;hostname;pwd    # Ausgabe des Datums, des Hostnamens und des Arbeitsverzeichnisses
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate    # virtuelle Umgebung aktivieren
# cpu-epyc-1
# gpu-v100-1
# gpu-v100-3
# gpu-a100-1

srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'insert_embeddings.py' -m 'doc2vec' -p 1 -a 'http://cpu-epyc-5.ies.uni-kassel.de:9200' -d '/mnt/datasets/Bahamas/SAC/0/*.pdf' -D '/mnt/stud/home/kgutekunst/visualizations/images/'