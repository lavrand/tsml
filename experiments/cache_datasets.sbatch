#!/bin/bash
#SBATCH --partition=main            # Specify the partition to use
#SBATCH --job-name=cache_datasets   # Name of the job
#SBATCH --output=cache_datasets_%j.log  # Output log file (%j will be replaced by the job ID)
#SBATCH --error=cache_datasets_%j.err   # Error log file
#SBATCH --time=01:00:00            # Set a maximum runtime (e.g., 1 hour)
#SBATCH --mem=8G                   # Memory required for the job
#SBATCH --cpus-per-task=4          # Number of CPU cores

# Load necessary modules (if required)
module load anaconda               # Load Anaconda if required
source activate tsml               # Activate your Python environment

# Run the Python script to cache datasets
python3 cache_datasets.py
