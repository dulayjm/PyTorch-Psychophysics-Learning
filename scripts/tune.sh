#!/bin/bash

#$ -N optuna-tune
#$ -q gpu
#S -M jdulay@nd.edu
#S -m abe
#$ -l gpu=4

BASE_PATH="$HOME/psychophysics-loss"
module load python

python "$BASE_PATH/tune.py" --dataset_file="processed_out_acc.csv" \
  --loss_fn="psych-rt" \
  --use_neptune=True
