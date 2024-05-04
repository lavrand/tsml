#!/bin/bash

# Bash script to run main.py with configuration files listed in configs.ini inside _configs folder

LOGFILE="script_log.txt"
CONFIGS_FILE="_configs/configs.ini"
echo "Starting script at $(date)" > $LOGFILE

# Read each line in configs.ini and use it as a config file name
while IFS= read -r config || [[ -n "$config" ]]; do
   echo "Starting main.py with $config at $(date)" >> $LOGFILE
   python main.py _configs/$config &
   echo "Started main.py with $config at $(date)" >> $LOGFILE
done < "$CONFIGS_FILE"

wait # Wait for all background processes to finish

echo "Script finished at $(date)" >> $LOGFILE
