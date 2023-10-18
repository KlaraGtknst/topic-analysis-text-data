#!/bin/bash      
#SBATCH --partition=main        # Partition main
#SBATCH --job-name=create-docs  # Job-Name
#SBATCH --nodes=1       # 1 Node wird benötigt
#SBATCH --cpus-per-task=128 
#SBATCH --nodelist=cpu-epyc-7
#SBATCH --mem=280g
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/%j.out        # Datei für stdout (logs/ prints != results, e.g., .pdf files) 
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err     # Datei für stderr
# (%N: Nodename, %j: Job-Nr.)

date;hostname;pwd    # Ausgabe des Datums, des Hostnamens und des Arbeitsverzeichnisses
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate    # virtuelle Umgebung aktivieren

srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'create_documents.py' -p 128 -a 'http://cpu-epyc-7.ies.uni-kassel.de:9200' -d '/mnt/datasets/Bahamas/*.pdf' -D '/mnt/stud/home/kgutekunst/visualizations/images/'