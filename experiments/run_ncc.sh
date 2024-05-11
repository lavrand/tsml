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
    {
        echo "Starting Python script for dataset: $dataset at $(date)" >> "experiment_ncc_${dataset}_${2}_${3}_${timestamp}.log"
        python3 ncc.py --datasets $dataset --metric $2 --gamma $3 >> "experiment_ncc_${dataset}_${2}_${3}_${timestamp}.log" 2>&1
        echo "Finished Python script for dataset: $dataset at $(date)" >> "experiment_ncc_${dataset}_${2}_${3}_${timestamp}.log"
    } || {
        echo "Python script failed for dataset: $dataset, metric: $2, gamma: $3"
    }
done