# Psychophysics-loss

The paper associated with this code will be published 
in the future. 

### Usage

1. Install Depedencies

```
python3 venv -m env
source env/bin/activate
pip3 install -r requirements.txt
```

2. Get files

You will need `.csv` files and directories from these links in order to run the larger experiments:

TODO: upload google drive links ...

2. Run via scripts

If you are using the crc, simpling run one of the scripts. `train.sh` is the standard
best run of reaction time as a pyschophysical paramerter. `cross-entropy.sh` deals with 
regular cross entropy loss as a control group, and `train_acc.sh` runs with accuracy as
a parameter. 

First, you made me to set your path like: 
```
echo $BASE_PATH
```
and format this directory and subdirectories before the working directory that you cloned for this repo. 

Then, you can call any script like: 
```
bash scripts/train.sh
```

3. Run manually

If you are running locally, you may also execute this command (or something similar), and change any of the CLI 
arguments: 

```
python "$BASE_PATH/main.py" --num_epochs=20  --dataset_file="small_dataset.csv"
```

4. Utilize Neptune

This experiment uses Neptune for logging. To utilize this feature, set the flag `use-neptune` to 
`True` and follow their [instructions](https://neptune.ai/) to export your custom API key. 