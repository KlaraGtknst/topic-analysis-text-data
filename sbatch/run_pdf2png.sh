#!/bin/bash      
#SBATCH --partition=main        # Partition main/ jupyter
#SBATCH --job-name=pdf2png-bahamas      # Job-Name
#SBATCH --nodes=1       # 1 Node wird benötigt
#SBATCH --cpus-per-task=20
#SBATCH --mem=256gb       # 100 GB Hauptspeicher
#SBATCH --time=56:00:00  # max. Laufzeit 56h
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/%j.out        # Datei für stdout (logs/ prints != results, e.g., .pdf files) 
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err     # Datei für stderr
# (%N: Nodename, %j: Job-Nr.)

date;hostname;pwd    # Ausgabe des Datums, des Hostnamens und des Arbeitsverzeichnisses
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate    # virtuelle Umgebung aktivieren
srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main_server.py 'convert_pdf2image.py' -d '/mnt/datasets/Bahamas' -o '/mnt/stud/home/kgutekunst/visualizations/images/'  # Programm ausfuehren
#srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/main.py 'convert_pdf2image.py' -d '/mnt/datasets/Bahamas' -o '/mnt/stud/home/kgutekunst/visualizations/images/'  # Programm ausfuehren
#srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/doc_images/convert_pdf2image.py -d '/mnt/datasets/Bahamas' -o '/mnt/stud/home/kgutekunst/visualizations/images/'  # Programm ausfuehren