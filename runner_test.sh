#!/bin/bash
#SBATCH --job-name=test-disksimtool
#SBATCH --nodes=4 --ntasks-per-node=16
#SBATCH --partition=short
#SBATCH --time=60
#SBATCH --mail-user=riccardo.franceschi@obspm.fr
#SBATCH --mail-type=BEGIN,END
#SBATCH --mem=10gb
#SBATCH --tmp=10gb

source /obs/rfranceschi/miniconda3/etc/profile.d/conda.sh
conda activate astromodels

SCRATCH=/scratch/$USER/run.${SLURM_JOBID}
srun --ntasks=$SLURM_JOB_NUM_NODES mkdir -p $SCRATCH
cd $SCRATCH
srun --ntasks=$SLURM_JOB_NUM_NODES cp /obs/$USER/run_fitter.py .
srun --ntasks=$SLURM_JOB_NUM_NODES cp /obs/$USER/menu_model.py .
srun --ntasks=$SLURM_JOB_NUM_NODES cp /data/$USER/profiles .
srun --ntasks=$SLURM_JOB_NUM_NODES cp /data/$USER/opacities .

mpiexec ./run_fitter.py > run_fitter.out
srun --ntasks=$SLURM_JOB_NUM_NODES mv myanalysis /data/$USER/

cd ${SLURM_SUBMIT_DIR}
mv ${SCRATCH}/run_fitter.out .
srun --ntasks=$SLURM_JOB_NUM_NODES rm -rf ${SCRATCH}

exit 0
