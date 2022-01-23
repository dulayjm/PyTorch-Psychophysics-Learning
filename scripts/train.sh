#!/bin/bash

#$ -N train-rt-20epoch-seed
#$ -q gpu
#S -m abe
#$ -l gpu=4

BASE_PATH="$HOME/PyTorch-Psychophysics-Learning" 
# module load python

python "$BASE_PATH/main.py" --num_epochs=20  --dataset_file="small_dataset.csv" --use_neptune=False
