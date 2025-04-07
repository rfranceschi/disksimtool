#!/bin/bash
#SBATCH --job-name=disksimtool
#SBATCH --nodes=4 --ntasks-per-node=16
#SBATCH --partition=long
#SBATCH --time=7200
#SBATCH --mail-user=riccardo.franceschi@obspm.fr
#SBATCH --mail-type=BEGIN,END
#SBATCH --mem=40gb
#SBATCH --tmp=40gb

export OMP_NUM_THREADS=1

source /obs/rfranceschi/miniconda3/etc/profile.d/conda.sh
conda activate astromodels

SCRATCH=/scratch/$USER/run.${SLURM_JOBID}
DATA=/data/$USER/TWHya

srun --ntasks=$SLURM_JOB_NUM_NODES mkdir -p $SCRATCH
cd $SCRATCH

srun --ntasks=${SLURM_JOB_NUM_NODES} cp ${SLURM_SUBMIT_DIR}/run_fitter.py .
srun --ntasks=$SLURM_JOB_NUM_NODES cp ${SLURM_SUBMIT_DIR}/menu_model.py .
srun --ntasks=$SLURM_JOB_NUM_NODES cp -r ${DATA}/opacities .
srun --ntasks=$SLURM_JOB_NUM_NODES cp -r ${DATA}/profiles .

mpiexec -n 16 python3 ${SCRATCH}/run_fitter.py > run_fitter.out
srun --ntasks=$SLURM_JOB_NUM_NODES mv corner.png /data/$USER/"corner_$RANDOM.out"
srun --ntasks=$SLURM_JOB_NUM_NODES mv myanalysis /data/$USER/"myanalysis_$RANDOM"

cd ${SLURM_SUBMIT_DIR}
mv ${SCRATCH}/run_fitter.out "run_fitter_$RANDOM.out"
srun --ntasks=$SLURM_JOB_NUM_NODES rm -rf ${SCRATCH}

exit 0
