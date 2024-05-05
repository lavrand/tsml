import os

# Get a list of all sbatch files in the current directory
sbatch_files = [f for f in os.listdir('.') if f.endswith('.sbatch')]

# Submit each sbatch file to the cluster
for sbatch_file in sbatch_files:
    os.system(f'sbatch {sbatch_file}')