import configparser
import os
import signal
import subprocess
import sys
from datetime import datetime

def log_message(message, log_file):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_file.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}\n")

# Check if a command line argument has been provided
if len(sys.argv) < 2:
    log_message("Usage: python main.py <config_file>")
    sys.exit(1)

# The second command line argument is expected to be the config file name
config_file = sys.argv[1]

# Directory paths
experiments_dir = 'experiments'
configs_dir = '_configs'

# Initialize the configparser
config = configparser.ConfigParser()

log_file_path = config.get('DEFAULT', 'log_file_path')
log_file = open(log_file_path, 'a')

log_file_error = open("error_log.txt", 'a')

PROCESS_TIMEOUT_SECONDS = config.getint('DEFAULT', 'PROCESS_TIMEOUT_SECONDS')

# Get a list of all Python files in the experiments directory
experiment_files = [f for f in os.listdir(experiments_dir) if f.endswith('.py')]

# Get a list of all config files in the configs directory
config_files = [f for f in os.listdir(configs_dir) if f.endswith('.ini')]

# Function to run the dispscript commands
def run_subprocess(command, i=None):
    stdout, stderr = None, None  # Initialize these to avoid UnboundLocalError

    # Start the process in a new session
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               preexec_fn=os.setsid)
    try:
        # communicate() waits for the process to complete or for the timeout to expire
        stdout, stderr = process.communicate(timeout=PROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        # If the timeout expires, kill the entire process group
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # try to terminate the process group gracefully
        process.wait(timeout=PROCESS_TIMEOUT_SECONDS)  # give it 10 seconds to terminate gracefully
        if process.poll() is None:  # if the process is still running after 10 seconds
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)  # forcibly kill the process group
        log_message(f"Command took longer than timeout seconds and was killed!", log_file)
    else:
        if process.returncode != 0:
            log_message(f"Command failed!", log_file)
        else:
            log_message(f"Command completed successfully!", log_file)

    # Return stdout, stderr, and returncode
    return stdout, stderr, process.returncode

# For each experiment file
for experiment_file in experiment_files:
    # For each config file
    for config_file in config_files:
        # Read the config file
        config = configparser.ConfigParser()
        config.read(os.path.join(configs_dir, config_file))

        # Prepare the command to run the experiment file with the config file as an argument
        command = ['python', os.path.join(experiments_dir, experiment_file), os.path.join(configs_dir, config_file)]

        # Run the command
        run_subprocess(' '.join(command))

log_file.close()
log_file_error.close()


