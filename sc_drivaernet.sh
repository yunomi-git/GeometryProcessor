#!/bin/bash

source /etc/profile
module load anaconda/2023b

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

python -m dataset.drivaernet_process_supercloud $LLSUB_RANK $LLSUB_SIZE