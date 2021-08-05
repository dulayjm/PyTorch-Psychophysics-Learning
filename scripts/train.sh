#!/bin/bash

#$ -N train-rt-20epoch-comparative
#$ -q gpu
#S -M jdulay@nd.edu
#S -m abe
#$ -l gpu=4

BASE_PATH="$HOME/psychophysics-loss"
module load python

python "$BASE_PATH/main.py" --num_epochs=20  --dataset_file="processed_out_acc.csv" --use_neptune=True
