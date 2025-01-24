#!/bin/bash
#SBATCH --partition main                 # Specify the partition to use
#SBATCH --time 6-23:50:00               # Set a maximum runtime (6 days, 23 hours, 50 minutes)
#SBATCH --job-name cache_datasets       # Name of the job
#SBATCH --output /cs_storage/andreyl/dtw_logs/cache_datasets-%J.out  # Output log file (%J will be replaced by the job ID)
#SBATCH --error /cs_storage/andreyl/dtw_error_logs/cache_datasets-%J.err   # Error log file (%J will be replaced by the job ID)
#SBATCH --mail-user=andreyl@post.bgu.ac.il  # Email for notifications
#SBATCH --mail-type=FAIL                # Notify only on failures
#SBATCH --mem=48G                       # Memory required for the job
#SBATCH --cpus-per-task=16               # Number of CPU cores
#SBATCH --tasks=1                       # Single task job

# Load necessary modules (if required)
module load anaconda                    # Load Anaconda if it's required
source activate tsml                   # Activate your Python environment

# Print some debug information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Caching datasets..."

# Run the Python script to cache datasets
python3 cache_datasets.py

# Print completion message
echo "Job completed at: $(date)"
