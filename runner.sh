#!/bin/bash
#SBATCH --job-name=disksimtool
#SBATCH --nodes=4 --ntasks-per-node=16
#SBATCH --partition=long
#SBATCH --time=7200
#SBATCH --mem=60gb
#SBATCH --tmp=60gb
#SBATCH --signal=B:SIGTERM@1800

SCRATCH=/scratch2/$USER/run.${SLURM_JOBID}
DATA=/data/$USER/TWHya
BACKUP=$DATA/runs/run.${SLURM_JOBID}

function debug_log() {
    echo "[`date +"%Y-%m-%d %H:%M:%S"`] $1"
}

function do_cleanup() {
    debug_log "Starting cleanup process..."

    if [[ -n "$BACKUP_PID" ]] && kill -0 $BACKUP_PID 2>/dev/null; then
      debug_log "Killing backup background process ($BACKUP_PID)"
      kill $BACKUP_PID
      wait $BACKUP_PID 2>/dev/null
    fi

    NODE_NAME=$(hostname)  # This gets the node name the job is running on
    JOBID=${SLURM_JOB_ID}

    # Move corner.png and update symlink
    if [[ -f corner.png ]]; then
        OUTPUT_NAME="corner_.out"
        mv corner.png ${SLURM_SUBMIT_DIR}/$OUTPUT_NAME
        ln -sf $OUTPUT_NAME ${SLURM_SUBMIT_DIR}/latest_corner.out
        debug_log "Moved corner.png and updated symlink"
    fi

    # Move myanalysis directory and update symlink
    if [[ -d myanalysis ]]; then
        DIR_NAME="myanalysis"
        mv myanalysis ${SLURM_SUBMIT_DIR}/$DIR_NAME
        ln -sf $DIR_NAME ${SLURM_SUBMIT_DIR}/latest_myanalysis
        debug_log "Moved myanalysis and updated symlink"
    fi

    # Move run_fitter.out and update symlink
    cd ${SLURM_SUBMIT_DIR}
    if [[ -f ${SCRATCH}/run_fitter.out ]]; then
        FILE_NAME="run_fitter.out"
        cp ${SCRATCH}/run_fitter.out ${SLURM_SUBMIT_DIR}/$FILE_NAME
        ln -sf $FILE_NAME ${SLURM_SUBMIT_DIR}/latest_run_fitter.out
        debug_log "Copied run_fitter.out and updated symlink"
    fi

    # Remove the scratch directory
    rm -rf ${SCRATCH}
    debug_log "Deleted scratch directory"
}

function sig_handler_SIGTERM() {
    debug_log "SIGTERM received — calling cleanup"
    cd $SCRATCH
    do_cleanup
    exit 2
}

trap 'sig_handler_SIGTERM' SIGTERM

export OMP_NUM_THREADS=1

debug_log "Activating conda environment"
source /obs/rfranceschi/miniconda3/etc/profile.d/conda.sh
conda activate astromodels

debug_log "Creating backup directory: $BACKUP"
mkdir -p $BACKUP
debug_log "Creating scratch directory: $SCRATCH"
mkdir -p $SCRATCH
cd $SCRATCH

debug_log "Copying input files..."
cp ${SLURM_SUBMIT_DIR}/run_fitter.py .
cp ${SLURM_SUBMIT_DIR}/menu_model.py .
cp -r ${DATA}/opacities .
cp -r ${DATA}/profiles .

debug_log "Starting background process to save files every 6 hours"
# This will periodically save files every 6 hours (21600 seconds) in the background
(
  while true; do
    sleep 21600  # Wait for 6 hours
    debug_log "Saving files periodically..."
    if [[ -d myanalysis ]]; then
      # Check if backup exists, delete if it does
        if [[ -d $BACKUP/myanalysis ]]; then
            rm -rf $BACKUP/myanalysis  # Remove the existing backup
            debug_log "Deleted existing backup of myanalysis"
        fi
        # Copy the new myanalysis directory
        rsync -a --delete myanalysis/ $BACKUP/myanalysis/
        debug_log "Copied myanalysis to backup directory"
    else
        debug_log "myanalysis directory does not exist yet, skipping backup"
    fi
  done
) &
BACKUP_PID=$!
export BACKUP_PID

debug_log "Starting run_fitter.py with mpiexec"
mpiexec -n 64 python3 ${SCRATCH}/run_fitter.py > run_fitter.out &
wait

debug_log "Main process complete — calling cleanup"
do_cleanup

debug_log "Job finished"
exit 0
