#!/bin/bash

source /etc/profile
module load anaconda/2023b

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

python -m shape_regression.generate_op_cache_diffusionnet --task_id $LLSUB_RANK --num_tasks $LLSUB_SIZE --input_folder DrivAerNet/train/