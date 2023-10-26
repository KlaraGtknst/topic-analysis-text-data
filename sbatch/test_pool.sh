#!/bin/bash      
#SBATCH --partition=main        # Partition main
#SBATCH --job-name=test-pool  # Job-Name
#SBATCH --nodes=1       # 1 Node wird benötigt
#SBATCH --cpus-per-task=90
#SBATCH --nodelist=cpu-epyc-1
#SBATCH --mem=300g
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/%j.out        # Datei für stdout (logs/ prints != results, e.g., .pdf files) 
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err     # Datei für stderr
# (%N: Nodename, %j: Job-Nr.)

date;hostname;pwd    # Ausgabe des Datums, des Hostnamens und des Arbeitsverzeichnisses
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate    # virtuelle Umgebung aktivieren

srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'test_pool.py' -p 90 -d '/mnt/datasets/Bahamas/*.pdf' -D '/mnt/stud/home/kgutekunst/visualizations/images/'
