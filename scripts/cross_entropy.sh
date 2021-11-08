#!/bin/bash

#$ -N train_cross_entropy_20epoch-seed
#$ -q gpu
#S -m abe
#$ -l gpu=4

BASE_PATH="$HOME/psychophysics-loss"
module load python

python "$BASE_PATH/main.py" --dataset_file="small_dataset.csv" \
  --loss_fn="cross-entropy" \
  --use_neptune=False
