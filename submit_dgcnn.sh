#!/bin/bash

source /etc/profile
module load anaconda/2023b

echo id $LLSUB_RANK
echo num proc: $LLSUB_SIZE

python -m heuristic_prediction.dgcnn_train