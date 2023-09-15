#!/bin/bash      
#SBATCH --partition=main        # Partition main
#SBATCH --job-name=init-db  # Job-Name
#SBATCH --nodes=1       # 1 Node wird benötigt
#SBATCH --cpus-per-task=32
#SBATCH --mem=264g       # 64 GB Hauptspeicher
#SBATCH --time=32:00:00  # max. Laufzeit 16h
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/%j.out        # Datei für stdout (logs/ prints != results, e.g., .pdf files) 
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err     # Datei für stderr
# (%N: Nodename, %j: Job-Nr.)

date;hostname;pwd    # Ausgabe des Datums, des Hostnamens und des Arbeitsverzeichnisses
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate    # virtuelle Umgebung aktivieren
srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'db_elasticsearch.py' -d '/mnt/datasets/Bahamas/SAC/0/*.pdf' -D '/mnt/stud/home/kgutekunst/visualizations/pdf2png'  # Programm ausfuehren
#srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/db_elasticsearch.py -d '/mnt/datasets/Bahamas' -D '/mnt/stud/home/kgutekunst/visualizations/images/'  # Programm ausfuehren