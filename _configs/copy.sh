#!/bin/bash

# Define the files to be copied
file1="run.sh"
file2="_update_configs_args.py"

# Define the destination directory (two levels up)
dest_dir="../../"

# Copy the files
cp "$file1" "$dest_dir"
cp "$file2" "$dest_dir"

echo "Files copied to $dest_dir"
