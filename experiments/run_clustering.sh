#!/bin/bash

# Activate the Conda environment
source activate new_env3

# Split the input string into an array using '_' as the delimiter
IFS='_' read -ra datasets <<< "$1"

# Run the Python script and redirect stdout and stderr to a unique log file
for dataset in "${datasets[@]}"
do
    # Get the current timestamp for each run
    timestamp=$(date +%Y%m%d%H%M%S)
    if [ "$3" == "softdtw" ]; then
        echo "Starting Python script for dataset: $dataset at $(date)" >> "experiment_clustering_${dataset}_${2}_${3}_${4}_${timestamp}.log"
        python3 clustering.py --datasets $dataset --n_clusters $2 --metric $3 --gamma $4 >> "experiment_clustering_${dataset}_${2}_${3}_${4}_${timestamp}.log" 2>&1
        echo "Finished Python script for dataset: $dataset at $(date)" >> "experiment_clustering_${dataset}_${2}_${3}_${4}_${timestamp}.log"
    else
        echo "Starting Python script for dataset: $dataset at $(date)" >> "experiment_clustering_${dataset}_${2}_${3}_${timestamp}.log"
        python3 clustering.py --datasets $dataset --n_clusters $2 --metric $3 >> "experiment_clustering_${dataset}_${2}_${3}_${timestamp}.log" 2>&1
        echo "Finished Python script for dataset: $dataset at $(date)" >> "experiment_clustering_${dataset}_${2}_${3}_${timestamp}.log"
    fi
done