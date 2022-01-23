#!/bin/bash

#$ -N eval_psych
#$ -q gpu
#S -m abe
#$ -l gpu=4

BASE_PATH="$HOME/PyTorch-Psychophysics-Learning"
# module load python

python "$BASE_PATH/test.py"  --dataset_file="small_dataset.csv" --use_neptune=False
