#!/bin/bash      
#SBATCH --partition=main        # Partition main/ jupyter
#SBATCH --job-name=pdf2png-bahamas      # Job-Name
#SBATCH --nodes=1       # 1 Node wird benötigt
#SBATCH --cpus-per-task=20
#SBATCH --mem=100gb       # 100 GB Hauptspeicher
#SBATCH --time=16:00:00  # max. Laufzeit 5 min
#SBATCH --output=/mnt/stud/home/kgutekunst/visualizations/%j.out        # Datei für stdout
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err     # Datei für stderr
# (%N: Nodename, %j: Job-Nr.)

date;hostname;pwd    # Ausgabe des Datums, des Hostnamens und des Arbeitsverzeichnisses
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate    # virtuelle Umgebung aktivieren
srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/convert_pdf2image.py -d '/mnt/datasets/Bahamas/SAC/0/*.pdf' -o '/mnt/stud/home/kgutekunst/visualizations/images/'  # Programm ausfuehren