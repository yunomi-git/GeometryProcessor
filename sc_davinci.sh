#!/bin/bash

source /etc/profile
module load anaconda/2023b

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

python -m dataset.davinci_process_supercloud --task_id $LLSUB_RANK --num_tasks $LLSUB_SIZE
