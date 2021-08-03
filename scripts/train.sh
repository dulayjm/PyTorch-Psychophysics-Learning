#!/bin/bash

#$ -N train_psych
#$ -q gpu
#S -M jdulay@nd.edu
#S -m abe
#$ -l gpu=4

BASE_PATH="$HOME/psychophysics-loss"
module load python

python "$BASE_PATH/main.py" --dataset_file="processed_out.csv" \
  --use_neptune=True
