#!/bin/bash

# Activate the Conda environment
source activate new_env2

# Get the current timestamp
timestamp=$(date +%Y%m%d%H%M%S)

# Run the Python script and redirect stdout and stderr to a unique log file
python3 ncc.py --dataset $1 --metric $2 --gamma $3 > "experiment_ncc_${1}_${2}_${3}_${timestamp}.log" 2>&1