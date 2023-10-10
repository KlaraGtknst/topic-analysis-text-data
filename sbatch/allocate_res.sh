#!/bin/bash      
#SBATCH --partition=main        # Partition main
#SBATCH --job-name=allocate-res  # Job-Name
#SBATCH --nodes=1       # 1 Node wird benötigt
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=cpu-epyc-1
#SBATCH --mem=64g       # 64 GB Hauptspeicher
#SBATCH --time=16:00:00  # max. Laufzeit 16h
#SBATCH --output=/mnt/stud/home/kgutekunst/logs/%j.out        # Datei für stdout (logs/ prints != results, e.g., .pdf files) 
#SBATCH --error=/mnt/stud/home/kgutekunst/error_logs/%j.err     # Datei für stderr
# (%N: Nodename, %j: Job-Nr.)


# gibt FEHLER: srun: error: ioctl(TIOCGWINSZ): Inappropriate ioctl for device
# srun: error: Not using a pseudo-terminal, disregarding --pty option
# WARNING: 'UsePAM no' is not supported in RHEL and may cause several problems.
# main: sshd: ssh-rsa algorithm is disabled

date;hostname;pwd    # Ausgabe des Datums, des Hostnamens und des Arbeitsverzeichnisses
source /mnt/stud/work/kgutekunst/bsc-py/bin/activate    # virtuelle Umgebung aktivieren
srun --pty /usr/sbin/sshd -D -f ~/sshd/sshd_config