#!/bin/bash
#SBATCH --job-name=disksimtool
#SBATCH --nodes=4 --ntasks-per-node=16
#SBATCH --partition=long
#SBATCH --time=7200
#SBATCH --mail-user=riccardo.franceschi@obspm.fr
#SBATCH --mail-type=BEGIN,END
#SBATCH --mem=40gb
#SBATCH --tmp=40gb

source /obs/rfranceschi/miniconda3/etc/profile.d/conda.sh
conda activate astromodels

SCRATCH=/scratch/$USER/run.${SLURM_JOBID}
DATA=/data/$USER/TWHya

mkdir -p $SCRATCH
cd $SCRATCH

cp ${SLURM_SUBMIT_DIR}/run_fitter.py .
cp ${SLURM_SUBMIT_DIR}/menu_model.py .
cp ${DATA}/opacities .
cp ${DATA}/profiles .

mpiexec -np 64 python3 ./run_fitter.py > run_fitter.out
mv corner.png /data/$USER/
mv myanalysis /data/$USER/

cd ${SLURM_SUBMIT_DIR}
mv ${SCRATCH}/run_fitter.out .
rm -rf ${SCRATCH}

exit 0
