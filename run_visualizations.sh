#!/bin/bash      
#SBATCH --partition=main        # Partition main/ jupyter
#SBATCH --job-name=vis-text      # Job-Name
#SBATCH --nodes=1       # 1 Node wird benötigt
#SBATCH --tasks-per-node=1      # Anzahl Tasks/ CPUs
#SBATCH --mem=100       # 100 MB Hauptspeicher
#SBATCH --time=0:05:00  # max. Laufzeit 5 min
#SBATCH --output=/mnt/stud/home/kgutekunst/visualizations/%j.out        # Datei für stdout
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err     # Datei für stderr
# (%N: Nodename, %j: Job-Nr.)
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate    # virtuelle Umgebung aktivieren
srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/visualize_texts.py -i '/mnt/datasets/Bahamas/SAC/0/SAC2-12.pdf' -o '/mnt/stud/home/kgutekunst/visualizations/'    # Programm ausfuehren
echo "Hello from `hostname`"    # Programm ausfuehren