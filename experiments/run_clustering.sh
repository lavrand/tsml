#!/bin/bash

# Activate the Conda environment
source activate new_env2

# Get the current timestamp
timestamp=$(date +%Y%m%d%H%M%S)

# Run the Python script and redirect stdout and stderr to a unique log file
python3 clustering.py --dataset $1 --n_clusters $2 --metric $3 --gamma $4 > "experiment_clustering_${1}_${2}_${3}_${4}_${timestamp}.log" 2>&1