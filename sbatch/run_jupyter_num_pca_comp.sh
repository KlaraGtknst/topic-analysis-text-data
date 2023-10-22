#!/bin/bash      
#SBATCH --partition=jupyter        
#SBATCH --job-name=n_pca_comp
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=2
# --nodelist=gpu-v100-1
#SBATCH --mem=264g      
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/%j.out       
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err 
# (%N: Nodename, %j: Job-Nr.)

date;hostname;pwd    # Ausgabe des Datums, des Hostnamens und des Arbeitsverzeichnisses
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate    # virtuelle Umgebung aktivieren
srun python /mnt/stud/work/kgutekunst/topic-analysis-text-data/num_pca_comp.ipynb