#!/bin/bash

#$ -N train-vgg-psych-run-dropout
#$ -q gpu
#S -m abe
#$ -l gpu=4

BASE_PATH="$HOME/research/PyTorch-Psychophysics-Learning" 
module load python
source env/bin/activate
python3 "$BASE_PATH/train.py" \
    --model_name="vgg" \
    --num_epochs=20 \
    --loss_fn="psych-rt" \
    --dataset_file="sigma_dataset.csv" \
    --log=0
