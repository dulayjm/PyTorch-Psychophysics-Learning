#!/bin/bash

#$ -N eval_psych_ce
#$ -q gpu
#S -M jdulay@nd.edu
#S -m abe
#$ -l gpu=4

BASE_PATH="$HOME/psychophysics-loss"
module load python

python "$BASE_PATH/test.py"  --dataset_file="processed_out_acc.csv" --use_neptune=True
