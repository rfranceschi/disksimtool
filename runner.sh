#!/bin/bash
#SBATCH --job-name=disksimtool
#SBATCH --nodes=4 --ntasks-per-node=16
#SBATCH --partition=long
#SBATCH --time=7200
#SBATCH --mail-user=riccardo.franceschi@obspm.fr
#SBATCH --mail-type=BEGIN,END
#SBATCH --mem=40gb
#SBATCH --tmp=40gb
#SBATCH --signal=B:SIGTERM@1800

SCRATCH=/scratch/$USER/run.${SLURM_JOBID}
DATA=/data/$USER/TWHya

function debug_log() {
    echo "[`date +"%Y-%m-%d %H:%M:%S"`] $1"
}

function do_cleanup() {
    debug_log "Starting cleanup process..."

    NODE_NAME=$(hostname)  # This gets the node name the job is running on
    JOBID=${SLURM_JOB_ID}

    # Move corner.png and update symlink
    if [[ -f corner.png ]]; then
        OUTPUT_NAME="corner_${JOBID}_${NODE_NAME}.out"
        mv corner.png ${SLURM_SUBMIT_DIR}/$OUTPUT_NAME
        ln -sf $OUTPUT_NAME ${SLURM_SUBMIT_DIR}/latest_corner.out
        debug_log "Moved corner.png and updated symlink from $(hostname)"
    fi

    # Move myanalysis directory and update symlink
    if [[ -d myanalysis ]]; then
        DIR_NAME="myanalysis_${JOBID}_${NODE_NAME}"
        mv myanalysis ${SLURM_SUBMIT_DIR}/$DIR_NAME
        ln -sf $DIR_NAME ${SLURM_SUBMIT_DIR}/latest_myanalysis
        debug_log "Moved myanalysis and updated symlink from $(hostname)"
    fi

    # Move run_fitter.out and update symlink
    cd ${SLURM_SUBMIT_DIR}
    if [[ -f ${SCRATCH}/run_fitter.out ]]; then
        FILE_NAME="run_fitter_${JOBID}_${NODE_NAME}.out"
        cp ${SCRATCH}/run_fitter.out ${SLURM_SUBMIT_DIR}/$FILE_NAME
        ln -sf $FILE_NAME ${SLURM_SUBMIT_DIR}/latest_run_fitter.out
        debug_log "Copied run_fitter.out and updated symlink from $(hostname)"
    fi

    # Remove the scratch directory
    rm -rf ${SCRATCH}
    debug_log "Deleted scratch directory in $(hostname)"
}

function sig_handler_SIGTERM() {
    debug_log "SIGTERM received — calling cleanup"
    do_cleanup
    exit 2
}

trap 'sig_handler_SIGTERM' SIGTERM

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
