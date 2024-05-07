#!/bin/bash

# Activate the Conda environment
source activate new_env2

# Install tslearn
pip install tslearn
pip install h5py

# Run the Python script and redirect stdout and stderr to a unique log file
for dataset in $1
do
    # Get the current timestamp for each run
    timestamp=$(date +%Y%m%d%H%M%S)
    {
        echo "Starting Python script for dataset: $dataset at $(date)" >> "experiment_clustering_${dataset}_${2}_${3}_${4}_${timestamp}.log"
        python3 clustering.py --dataset $dataset --n_clusters $2 --metric $3 --gamma $4 >> "experiment_clustering_${dataset}_${2}_${3}_${4}_${timestamp}.log" 2>&1
        echo "Finished Python script for dataset: $dataset at $(date)" >> "experiment_clustering_${dataset}_${2}_${3}_${4}_${timestamp}.log"
    } || {
        echo "Python script failed for dataset: $dataset, n_clusters: $2, metric: $3, gamma: $4"
    }
done