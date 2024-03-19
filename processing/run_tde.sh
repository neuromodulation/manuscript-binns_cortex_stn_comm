#!/bin/sh
#SBATCH --ntasks=8
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=100G
#SBATCH --partition=medium
#SBATCH -o logs/slurm/slurm-%j-%a.out
#SBATCH -e logs/errors/error-%j-%a.out
#SBATCH -t 2-00:00:00

module load python
python data_processing_tde_preset_hpc.py $SLURM_ARRAY_TASK_ID $1 $2 $3
