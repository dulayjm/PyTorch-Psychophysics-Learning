#!/bin/bash

#$ -N optuna_tune_crossentropy
#$ -q gpu
#S -m abe
#$ -l gpu=4

BASE_PATH="$HOME/psychophysics-loss"
# module load python
# pip3 install -r requirements.txt

python "$BASE_PATH/tune.py" --dataset_file="small_dataset.csv.csv" \
  --loss_fn="cross-entropy" \
  --use_neptune=True
