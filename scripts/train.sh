#!/bin/bash

#$ -N train_psyphy
#$ -q gpu
#S -M jdulay@nd.edu
#S -m abe
#$ -l gpu=4

BASE_PATH="$HOME/psychophysics-loss"
module load python

python "$BASE_PATH/main.py" --dataset_file="processed.csv" --use_neptune=True
