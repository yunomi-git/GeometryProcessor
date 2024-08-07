#!/bin/bash

source /etc/profile
module load anaconda/2023b

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

python -m supercloud.supercloud_test 48 $LLSUB_RANK $LLSUB_SIZE