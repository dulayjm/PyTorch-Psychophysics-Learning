#!/bin/bash

#$ -N train_psyphy
#$ -q gpu
#S -M jdulay@nd.edu
#S -m abe
#$ -l gpu=1

BASE_PATH="$HOME/psychophysics-loss"
module load python

python "$BASE_PATH/main.py"