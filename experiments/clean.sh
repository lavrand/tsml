#!/bin/bash

# Get a list of all sbatch files in the current directory, excluding cache_datasets.sbatch
sbatch_files=$(ls *.sbatch | grep -v 'cache_datasets.sbatch')

# Remove each sbatch file
for sbatch_file in $sbatch_files; do
    rm "$sbatch_file"
    echo "Removed $sbatch_file"
done

echo "All .sbatch files except cache_datasets.sbatch have been removed."