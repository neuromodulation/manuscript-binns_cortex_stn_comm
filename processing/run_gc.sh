#!/bin/sh
#SBATCH --ntasks=8
#SBATCH -N 1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=80G
#SBATCH --partition=short
#SBATCH -o logs/slurm/slurm-%j-%a.out
#SBATCH -e logs/errors/error-%j-%a.out
#SBATCH -t 04:00:00

module load python
python data_processing_granger_causality_preset_hpc.py $SLURM_ARRAY_TASK_ID $1 $2 $3
