#!/bin/bash	 
#SBATCH --partition=main	# Partition main/ jupyter
#SBATCH --nodes=1	# 1 Node wird benötigt
#SBATCH --tasks-per-node=1	# Anzahl Tasks/ CPUs
#SBATCH --mem=100	# 100 MB Hauptspeicher
#SBATCH --time=0:05:00	# max. Laufzeit 5 min
#SBATCH --output=/mnt/stud/home/kgutekunst/visualizations/%j.out	# Datei für stdout
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err	# Datei für stderr
# (%N: Nodename, %j: Job-Nr.)
source bsc-py/bin/activate    # virtuelle Umgebung aktivieren
python3 visualizations.py	# Programm ausfuehren
echo "Hello from `hostname`"	# Programm ausfuehren