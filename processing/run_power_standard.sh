#!/bin/sh
#SBATCH --ntasks=10
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=50G
#SBATCH --partition=short
#SBATCH -o logs/slurm/slurm-%j-%a.out
#SBATCH -e logs/errors/error-%j-%a.out
#SBATCH -t 04:00:00

module load python
python data_processing_power_standard_preset_hpc.py $SLURM_ARRAY_TASK_ID $1 $2 $3
