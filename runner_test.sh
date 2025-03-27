#!/bin/bash
#SBATCH --job-name=disksimtool_test
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --partition=medium
#SBATCH --time=240
#SBATCH --mail-user=riccardo.franceschi@obspm.fr
#SBATCH --mail-type=BEGIN,END
#SBATCH --mem=10gb
#SBATCH --tmp=10gb

source /obs/rfranceschi/miniconda3/etc/profile.d/conda.sh
conda activate astromodels

srun python menu_model.py
