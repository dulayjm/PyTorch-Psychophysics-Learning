#!/bin/bash

#$ -N optuna_tune
#$ -q gpu
#S -M jdulay@nd.edu
#S -m abe
#$ -l gpu=4

BASE_PATH="$HOME/psychophysics-loss"
module load python
pip3 install -r requirements.txt

python "$BASE_PATH/tune.py" --dataset_file="processed_out_acc.csv" \
  --loss_fn="psych-rt" \
  --use_neptune=True
