#!/bin/bash

# Read the configuration file
config=$(cat run_config.json)

# Extract the repository URL and the Python script name
repository_url=$(echo $config | jq -r '.repository_url')
python_script=$(echo $config | jq -r '.python_script')

# Initialize arrays to store job IDs and parameters
declare -A job_ids
declare -A final_parameters

# Loop through the jobs
for job in $(echo $config | jq -r '.jobs[] | @base64'); do
    # Decode the job
    job=$(echo $job | base64 --decode)

    # Extract the job parameters
    id=$(echo $job | jq -r '.id')
    param1=$(echo $job | jq -r '.param1')
    param2=$(echo $job | jq -r '.param2')
    param3=$(echo $job | jq -r '.param3')
    param4=$(echo $job | jq -r '.param4')

    echo "Cloning repository into directory $id"
    git clone $repository_url $id

    echo "Changing directory to $id/_configs"
    cd $id/_configs

    echo "Running Python script with parameters --param1=$param1 --param2=$param2 --param3=$param3 --param4=$param4"
    python3 $python_script --param1 $param1 --param2 $param2 --param3 $param3 --param4 $param4

    echo "Going one level up from $id/_configs"
    cd ..

    if [ -f "sbatch.sbatch" ]; then
        echo "Submitting job in directory $id"
        output=$(sbatch sbatch.sbatch)
        job_id=$(echo $output | grep -oP '(?<=Submitted batch job )\d+')
        job_ids[$id]=$job_id
        final_parameters[$id]="$param1,$param2,$param3,$param4"
        echo "$id - $job_id"
    else
        echo "sbatch.sbatch not found in directory $id"
    fi

    echo "Going back to the parent directory from $id"
    cd ..
done

echo "Script execution completed."
echo "id,batch,param1,param2,param3,param4"
for id in ${!job_ids[@]}; do
    echo "$id,${job_ids[$id]},${final_parameters[$id]}"
done