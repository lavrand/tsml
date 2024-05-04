#!/bin/bash

# Define the parameters for each folder
declare -A time_expansions=(
    [1]=10 [2]=25 [3]=50 [4]=100 [5]=200 [6]=300 [7]=500 [8]=1000
    [9]=10 [10]=25 [11]=50 [12]=100 [13]=200 [14]=300 [15]=500 [16]=1000
    [17]=10 [18]=25 [19]=50 [20]=100 [21]=200 [22]=300 [23]=500 [24]=1000
)

declare -A dispatch_threshold=(
    [1]=0.25 [2]=0.25 [3]=0.25 [4]=0.25 [5]=0.25 [6]=0.25 [7]=0.25 [8]=0.25
    [9]=0.1 [10]=0.1 [11]=0.1 [12]=0.1 [13]=0.1 [14]=0.1 [15]=0.1 [16]=0.1
    [17]=0.025 [18]=0.025 [19]=0.025 [20]=0.025 [21]=0.025 [22]=0.025 [23]=0.025 [24]=0.025
)

#Option 1
#declare -A subtree_focus_threshold=(
#  [1]=0.125 [2]=0.125 [3]=0.125 [4]=0.125 [5]=0.125 [6]=0.125 [7]=0.125 [8]=0.125
#  [9]=0.05 [10]=0.05 [11]=0.05 [12]=0.05 [13]=0.05 [14]=0.05 [15]=0.05 [16]=0.05
#  [17]=0.0125 [18]=0.0125 [19]=0.0125 [20]=0.0125 [21]=0.0125 [22]=0.0125 [23]=0.0125 [24]=0.0125
#)
#
##dispatch_frontier_size with values 10
#declare -A dispatch_frontier_size=(
#  [1]=10 [2]=10 [3]=10 [4]=10 [5]=10 [6]=10 [7]=10 [8]=10
#  [9]=10 [10]=10 [11]=10 [12]=10 [13]=10 [14]=10 [15]=10 [16]=10
#  [17]=10 [18]=10 [19]=10 [20]=10 [21]=10 [22]=10 [23]=10 [24]=10
#)

#Option 2
declare -A subtree_focus_threshold=(
  [1]=1 [2]=1 [3]=1 [4]=1 [5]=1 [6]=1 [7]=1 [8]=1
  [9]=1 [10]=1 [11]=1 [12]=1 [13]=1 [14]=1 [15]=1 [16]=1
  [17]=1 [18]=1 [19]=1 [20]=1 [21]=1 [22]=1 [23]=1 [24]=1
)

#dispatch_frontier_size with values 1
declare -A dispatch_frontier_size=(
  [1]=1 [2]=1 [3]=1 [4]=1 [5]=1 [6]=1 [7]=1 [8]=1
  [9]=1 [10]=1 [11]=1 [12]=1 [13]=1 [14]=1 [15]=1 [16]=1
  [17]=1 [18]=1 [19]=1 [20]=1 [21]=1 [22]=1 [23]=1 [24]=1
)



# Initialize arrays to store job IDs and parameters
declare -A job_ids
declare -A final_parameters

# Loop through the folders
for i in {1..24}
do
    echo "Cloning repository into directory $i"
    # Clone the repository into a folder named as the current number
    git clone https://github.com/lavrand/ai-planners.git $i

    echo "Changing directory to $i/_configs"
    # Change directory to _configs within the cloned repository
    cd $i/_configs

    echo "Running Python script with parameters --dispatch-frontier-size=${dispatch_frontier_size[$i]} --subtree-focus-threshold=${subtree_focus_threshold[$i]} --time-expansions=${time_expansions[$i]} --dispatch-threshold=${dispatch_threshold[$i]} "
    # Run the python script with the specified parameters
    python3 _update_configs_args.py --dispatch-frontier-size ${dispatch_frontier_size[$i]} --subtree-focus-threshold ${subtree_focus_threshold[$i]} --time-expansions ${time_expansions[$i]} --dispatch-threshold ${dispatch_threshold[$i]}

    # Go one level up before running sbatch
    echo "Going one level up from $i/_configs"
    cd ..

    # Checking if sbatch.sbatch exists in the current directory
    if [ -f "sbatch.sbatch" ]; then
        echo "Submitting job in directory $i"
        # Submit the job and capture the output
        output=$(sbatch sbatch.sbatch)

        # Extract the job ID from the output
        job_id=$(echo $output | grep -oP '(?<=Submitted batch job )\d+')

        # Store the job ID and parameters in the arrays
        job_ids[$i]=$job_id
        final_parameters[$i]="${time_expansions[$i]},${subtree_focus_threshold[$i]},${dispatch_threshold[$i]},${dispatch_frontier_size[$i]}"

        # Print the mapping of folder number to job ID
        echo "$i - $job_id"
    else
        echo "sbatch.sbatch not found in directory $i"
    fi

    # Go back to the parent directory
    echo "Going back to the parent directory from $i"
    cd ..
done

# Log that the script has completed and display all job IDs in CSV format
echo "Script execution completed."
echo "id,batch,EPS,subtree_focus_threshold,dispatch_threshold,dispatch_frontier_size"
for i in {1..24}
do
    echo "$i,${job_ids[$i]},${final_parameters[$i]}"
done
