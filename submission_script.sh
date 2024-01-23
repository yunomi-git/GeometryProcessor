#!/bin/bash

source /etc/profile
module load anaconda/2023b

echo id $LLSUB_RANK
echo num proc: $LLSUB_SIZE

python -m supercloud_test 20 $LLSUB_RANK $LLSUB_SIZE