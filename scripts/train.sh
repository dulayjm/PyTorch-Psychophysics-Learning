#!/bin/bash

#$ -N train_psyphy
#$ -q gpu-debug
#S -M jdulay@nd.edu
#S -m abe
#$ -l gpu=4

BASE_PATH="$HOME/psychophysics-loss"
module load python

python "$BASE_PATH/main.py" --use_neptune=True
