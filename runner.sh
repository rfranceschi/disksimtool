#!/bin/bash
#SBATCH --job-name=disksimtool
#SBATCH --nodes=4 --ntasks-per-node=16
#SBATCH --partition=long
#SBATCH --time=7200
#SBATCH --mail-user=riccardo.franceschi@obspm.fr
#SBATCH --mail-type=BEGIN,END
#SBATCH --mem=40gb
#SBATCH --tmp=40gb
#SBATCH --signal=B:SIGINT@900

SCRATCH=/scratch/$USER/run.${SLURM_JOBID}
DATA=/data/$USER/TWHya

function debug_log() {
    echo "[`date +"%Y-%m-%d %H:%M:%S"`] $1"
}

function do_cleanup() {
    debug_log "Starting cleanup process..."

    if [[ -f corner.png ]]; then
        srun --ntasks=$SLURM_JOB_NUM_NODES mv corner.png /data/$USER/"corner_$RANDOM.out"
        debug_log "Moved corner.png to /data"
    fi

    if [[ -d myanalysis ]]; then
        srun --ntasks=$SLURM_JOB_NUM_NODES mv myanalysis /data/$USER/"myanalysis_$RANDOM"
        debug_log "Moved myanalysis directory to /data"
    fi

    cd ${SLURM_SUBMIT_DIR}
    if [[ -f ${SCRATCH}/run_fitter.out ]]; then
        srun --ntasks=$SLURM_JOB_NUM_NODES cp ${SCRATCH}/run_fitter.out "run_fitter_$RANDOM.out"
        debug_log "Copied run_fitter.out to submission directory"
    fi

    srun --ntasks=$SLURM_JOB_NUM_NODES rm -rf ${SCRATCH}
    debug_log "Deleted scratch directory"
}

function sig_handler_SIGINT() {
    debug_log "SIGINT received — calling cleanup"
    do_cleanup
    exit 2
}

trap 'sig_handler_SIGINT' SIGINT

export OMP_NUM_THREADS=1

debug_log "Activating conda environment"
source /obs/rfranceschi/miniconda3/etc/profile.d/conda.sh
conda activate astromodels

debug_log "Creating scratch directory: $SCRATCH"
srun --ntasks=$SLURM_JOB_NUM_NODES mkdir -p $SCRATCH
cd $SCRATCH

debug_log "Copying input files..."
srun --ntasks=${SLURM_JOB_NUM_NODES} cp ${SLURM_SUBMIT_DIR}/run_fitter.py .
srun --ntasks=$SLURM_JOB_NUM_NODES cp ${SLURM_SUBMIT_DIR}/menu_model.py .
srun --ntasks=$SLURM_JOB_NUM_NODES cp -r ${DATA}/opacities .
srun --ntasks=$SLURM_JOB_NUM_NODES cp -r ${DATA}/profiles .

debug_log "Starting run_fitter.py with mpiexec"
mpiexec -n 64 python3 ${SCRATCH}/run_fitter.py > run_fitter.out &
wait

debug_log "Main process complete — calling cleanup"
do_cleanup

debug_log "Job finished"
exit 0