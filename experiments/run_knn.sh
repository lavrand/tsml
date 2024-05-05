#!/bin/bash

# Get the current timestamp
timestamp=$(date +%Y%m%d%H%M%S)

# Run the Python script and redirect stdout and stderr to a unique log file
python3 knn.py --dataset $1 --k $2 --metric $3 --gamma $4 > "experiment_knn_${1}_${2}_${3}_${4}_${timestamp}.log" 2>&1