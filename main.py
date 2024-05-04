import configparser
import itertools
import multiprocessing
import os
import signal
import subprocess
import sys
import tarfile
from datetime import datetime


# Define a logging function
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

# Initialize the configparser
config = configparser.ConfigParser()

log_file_error = open("error_log.txt", 'a')

# Read the configuration file passed as a command line argument
try:
    config.read(config_file)
except configparser.DuplicateOptionError as e:
    log_message(f"Duplicate option error in config file: {e}", log_file_error)


# Retrieve the values from the config file
PROCESS_TIMEOUT_SECONDS = config.getint('DEFAULT', 'PROCESS_TIMEOUT_SECONDS')

log_file_path = config.get('DEFAULT', 'log_file_path')
CPU_COUNT_PC = multiprocessing.cpu_count()
CPU_COUNT = CPU_COUNT_PC

# Flag to enable or disable parallel processing
log_file = open(log_file_path, 'a')


for current_pfile in pfile_values:  # Looping over the defined range
    try:  # Add a try block to catch any exceptions that occur for a single PFILE_N.

        log_message(
            f" Starting a new set of exp"
            f"eriments  = {PFILE_N} ...",
            log_file)

        def run_subprocess_args(command, args):
            """
            Run a command with arguments and handle exceptions.

            Parameters:
            command (str): The command to execute.
            args (list): A list of arguments for the command.
            """
            # Combine the command and its arguments into one list.
            cmd = [command] + args

            try:
                # Execute the command with arguments, and wait for it to complete, but not longer than PLAN_SEARCH_TIMEOUT_SECONDS * 2
                result = subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=PLAN_SEARCH_TIMEOUT_SECONDS * 2)

                # If the command was successful, result.stdout will contain the output
                log_message(result.stdout, log_file)

            except subprocess.TimeoutExpired:
                # Handle the timeout exception as you see fit
                log_message("The command did not complete within timeout seconds.", log_file)
                # Here you might choose to try the command again, or perhaps record the timeout in a log file

            except subprocess.CalledProcessError as e:
                # Handle the exception for a non-zero exit code if check=True
                log_message(f"The command failed because: {e.stderr}", log_file)
                # Here you can do additional handling of the error, like retrying the command or logging the error

            except Exception as e:
                log_message(f"An error occurred: {str(e)}", log_file)
                return None


        base_command_common = (
            f"./rewrite-no-lp --time-based-on-expansions-per-second {time_based_on_expansions_per_second} "
            f"--include-metareasoning-time --multiply-TILs-by {multiply_TILs_by} "
            f"--real-to-plan-time-multiplier {real_to_plan_time_multiplier} --calculate-Q-interval {calculate_Q_interval} "
            f"--add-weighted-f-value-to-Q {add_weighted_f_value_to_Q} --min-probability-failure {min_probability_failure} "
            f"--slack-from-heuristic --forbid-self-overlapping-actions "
            f"--deadline-aware-open-list {deadline_aware_open_list} --ijcai-gamma {ijcai_gamma} --ijcai-t_u {ijcai_t_u} "
            f"--icaps-for-n-expansions {icaps_for_n_expansions} --time-aware-heuristic {time_aware_heuristic} "
            f"--dispatch-frontier-size {dispatch_frontier_size} --subtree-focus-threshold {subtree_focus_threshold} "
            f"--dispatch-threshold {dispatch_threshold} --optimistic-lst-for-dispatch-reasoning "
        )

        base_command_end = (f" %s pfile{PFILE_N}" % DOMAIN)

        # Function to run the dispscript commands

        def run_subprocess(command, i=None):
            stdout, stderr = None, None  # Initialize these to avoid UnboundLocalError

            # Start the process in a new session
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       preexec_fn=os.setsid)
            try:
                # communicate() waits for the process to complete or for the timeout to expire
                stdout, stderr = process.communicate(timeout=PLAN_SEARCH_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                # If the timeout expires, kill the entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # try to terminate the process group gracefully
                process.wait(timeout=10)  # give it 10 seconds to terminate gracefully
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

        log_message(f" Finished current set of experiments. Starting archiving...", log_file)
        log_message(f" All experiments completed for this cycle. Restarting...", log_file)

    except Exception as e:  # Catch any type of exception
        log_message(f" An error occurred during processing : {str(e)}", log_file)
        log_message("Continuing with the next experiment...", log_file)
    finally:
        pass

log_file.close()
log_file_error.close()


