#!/bin/bash

# Activate the Conda environment
source activate new_env2

# Install tslearn
pip install tslearn

# Get the current timestamp
timestamp=$(date +%Y%m%d%H%M%S)

# Run the Python script and redirect stdout and stderr to a unique log file
for dataset in $1
do
    python3 knn.py --dataset $dataset --k $2 --metric $3 --gamma $4 > "experiment_knn_${dataset}_${2}_${3}_${4}_${timestamp}.log" 2>&1
done