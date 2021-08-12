#!/bin/bash

#$ -N train_cross_entropy_20epoch-seed
#$ -q gpu
#S -M jdulay@nd.edu
#S -m abe
#$ -l gpu=4

BASE_PATH="$HOME/psychophysics-loss"
module load python

python "$BASE_PATH/main.py" --dataset_file="processed_out_acc.csv" \
  --loss_fn="cross-entropy" \
  --use_neptune=True
