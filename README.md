# Psychophysics-loss

This repository walks through integrating a simple psychophysics parameters gathered 
from a 2AFC Amazon Turk Experiment. The full dataset will be released soon. For now, 
utilize the `small_acc.csv` for quick computations.

### Usage

1. Install Depedencies. 

```
python3 venv -m env
source env/bin/activate
pip3 install -r requirements.txt
```

2. Run via scripts

If you are using the crc, simpling run one of the scripts. `train.sh` is the standard
best run of reaction time as a pyschophysical paramerter. `cross-entropy.sh` deals with 
regular cross entropy loss as a control group, and `train_acc.sh` runs with accuracy as
a parameter. 

3. Run locally

If you are running locally, you may also execute this command, and change any of the CLI 
arguments: 

```
python "$BASE_PATH/main.py" --num_epochs=20  --dataset_file="small_acc.csv"
```

4. Utilize Neptune

This experiment uses Neptune for logging. To utilize this feature, set the flag `use-neptune` to 
`True` and follow their [instructions](https://neptune.ai/) to export your custom API key. 