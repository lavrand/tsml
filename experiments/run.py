import os

# Get a list of all sbatch files in the current directory, excluding cache_datasets.sbatch
sbatch_files = [f for f in os.listdir('.') if f.endswith('.sbatch') and f != 'cache_datasets.sbatch']

# Initialize a counter for the number of batches started
batches_started = 0

# Submit each sbatch file to the cluster
for sbatch_file in sbatch_files:
    os.system(f'sbatch {sbatch_file}')
    batches_started += 1

# Print the number of batches started
print(f"Number of batches started: {batches_started}")