#!/bin/bash

source /etc/profile
module load anaconda/2023b

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

python -m dataset.process_and_save_temp --task_id $LLSUB_RANK --num_tasks $LLSUB_SIZE --input_folder DrivAerNet/Simplified_Remesh/ --output_folder DrivAerNet/train/