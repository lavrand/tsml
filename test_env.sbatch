#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=6-23:50:00
#SBATCH --job-name=test_env
#SBATCH --output my_job-id-%J.out### output log for running job - %J is the job number variable
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4

### Print some data to output file ###
echo "SLURM_JOBID”=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST

# Load the Anaconda module
module load anaconda

# Initialize Conda
source activate tsml

# Debug: Verify the active Python path and environment
echo "Python path: $(which python)"
python test.py