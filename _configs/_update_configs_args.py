import math
import argparse

# Constants
TOTAL_INPUT = 100  # Total input value
N = 10              # Number of configuration files

# Function to calculate the range for each file
def calculate_ranges(total_input, num_files):
    range_size = math.ceil(total_input / num_files)
    return [(i * range_size + 1, min((i + 1) * range_size, total_input)) for i in range(num_files)]

# Create argument parser
parser = argparse.ArgumentParser(description='Update configuration files.')
parser.add_argument('--dispatch-frontier-size', type=int, default=10,
                    help='Dispatch frontier size (can be 1, 10)')
parser.add_argument('-s', '--subtree-focus-threshold', type=float, required=True)
parser.add_argument('--time-expansions', type=int, default=1000,
                    help='Time based on expansions per second (can be 10, 25, 50, 100, 200, 300, 500, 1000)')
parser.add_argument('--dispatch-threshold', type=float, default=0.025,
                    help='Dispatch threshold (can be 0.025, 0.1, 0.25)')

# Parse arguments
args = parser.parse_args()

# New constants from arguments
TIME_BASED_ON_EXPANSIONS_PER_SECOND = args.time_expansions
DISPATCH_THRESHOLD = args.dispatch_threshold
SUBTREE_FOCUS_THRESHOLD = args.subtree_focus_threshold
dispatch_frontier_size = args.dispatch_frontier_size

# Other constants
PLAN_SEARCH_TIMEOUT_SECONDS = 600
DOMAIN = "rcll_domain_production_durations_time_windows.pddl"

# Calculate ranges
ranges = calculate_ranges(TOTAL_INPUT, N)

# Update configuration files
for i, (start, end) in enumerate(ranges, 1):
    config_file_name = "config{}.ini".format(i)  # Compatible with older Python versions

    # Read existing content
    with open(config_file_name, 'r') as file:
        lines = file.readlines()

    # Update the relevant lines
    with open(config_file_name, 'w') as file:
        for line in lines:
            if line.strip().startswith('PFILE_START ='):
                file.write(f"PFILE_START = {start}\n")
            elif line.strip().startswith('PFILE_END ='):
                file.write(f"PFILE_END = {end}\n")
            elif line.strip().startswith('time_based_on_expansions_per_second ='):
                file.write(f"time_based_on_expansions_per_second = {TIME_BASED_ON_EXPANSIONS_PER_SECOND}\n")
            elif line.strip().startswith('PLAN_SEARCH_TIMEOUT_SECONDS ='):
                file.write(f"PLAN_SEARCH_TIMEOUT_SECONDS = {PLAN_SEARCH_TIMEOUT_SECONDS}\n")
            elif line.strip().startswith('DOMAIN ='):
                file.write(f"DOMAIN = {DOMAIN}\n")
            elif line.strip().startswith('subtree_focus_threshold ='):
                file.write(f"subtree_focus_threshold = {SUBTREE_FOCUS_THRESHOLD}\n")
            elif line.strip().startswith('dispatch_threshold ='):
                file.write(f"dispatch_threshold = {DISPATCH_THRESHOLD}\n")
            elif line.strip().startswith('dispatch_frontier_size ='):
                file.write(f"dispatch_frontier_size = {dispatch_frontier_size}\n")
            else:
                file.write(line)

print("Configuration files have been updated.")
