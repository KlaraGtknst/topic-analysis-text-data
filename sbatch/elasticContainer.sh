#!/bin/bash      
#SBATCH --partition=main        # Partition main
#SBATCH --job-name=elastic-db  # Job-Name
#SBATCH --nodes=1       # 1 Node wird benötigt
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g       # 64 GB Hauptspeicher
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/%j.out        # Datei für stdout (logs/ prints != results, e.g., .pdf files) 
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err     # Datei für stderr
# (%N: Nodename, %j: Job-Nr.)

date;hostname;pwd    # Ausgabe des Datums, des Hostnamens und des Arbeitsverzeichnisses
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate    # virtuelle Umgebung aktivieren
cd /mnt/stud/work/kgutekunst/topic-analysis-text-data/ies-server
srun podman-compose up