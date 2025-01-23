#!/bin/bash

module load anaconda
# Activate the Conda environment
source activate tsml

# Split the input string into an array using '_' as the delimiter
IFS='_' read -ra datasets <<< "$1"

# Run the Python script and redirect stdout and stderr to a unique log file
for dataset in "${datasets[@]}"
do
    # Get the current timestamp for each run
    timestamp=$(date +%Y%m%d%H%M%S)
    echo "Starting Python script for dataset: $dataset at $(date)" >> "experiment_shapedtw_knn_${dataset}_${2}_${timestamp}.log"
    python3 shapedtw_knn.py --dataset $dataset --shape_function $2 >> "experiment_shapedtw_knn_${dataset}_${2}_${timestamp}.log" 2>&1
    echo "Finished Python script for dataset: $dataset at $(date)" >> "experiment_shapedtw_knn_${dataset}_${2}_${timestamp}.log"
done