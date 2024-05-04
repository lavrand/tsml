#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like so ##SBATCH
#SBATCH --partition main ### specify partition name where to run a job. main - 7 days time limit
#SBATCH --time 6-23:50:00 ### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name ai_planner_job ### name of the job. replace my_job with your desired job name
#SBATCH --output /cs_storage/andreyl/pancake/ai_planner_job-id-%J.out ### output log for running job - %J is the job number variable
#SBATCH --mail-user=andreyl@post.bgu.ac.il ### users email for sending job status notifications
#SBATCH --mail-type=BEGIN,END,FAIL ### conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=24G
#SBATCH --cpus-per-task=10 # 6 cpus per task  use for multithreading, usually with --tasks=1
#SBATCH --tasks=1 # 2 processes  use for processing of few programs concurrently in a job (with srun). Use just 1 otherwise
### Print some data to output file ###
### Start you code below ####
### Start your code below ####
module load anaconda
source activate new_env

echo "Current working directory is:"
pwd
echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
output=$(./_run_configs.sh)
status=$?
if [ $status -ne 0 ]; then
    echo "Script failed with status $status"
fi

wait
