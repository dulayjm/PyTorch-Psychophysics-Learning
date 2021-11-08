# Psychophysically Informed Learning

The paper associated with this code will be published 
in the future. 🧠🖥️

## Dataset

The dataset is the `OmniglotReactionTimeDataset` class that appears in some of the files in this project. For this experiment, you will need: 

- `omniglot_realfake` psychophysically annotated data points.
- `sigma_dataset.csv`
- `OmniglotReactionTimeDataset` class

The first subfolders are all of the raw images that are needed for the usage of the class. The `real` subfolder is a subset of 100 classes from the full Omniglot Dataset. The `fake` folder are DCGAN generated approximations of the each of the same classes from the first folder. The generative images were used as a form of data augmentation to increase intraclass variance exposure to human subjects on the psychophysical experiments in the past. The data loader will load images from both. 

The csv file is simply a reference structure of the data folder to load more easily. Each consists of the two paired images used in a given task, as well as the reaction time on the task and mean accuracy per the real label. 

The first class is the dataset class, subclassed from the Pytorch `Dataset` class. The `__getitem__` function is the most important one. When called, it return a dictionary like: 
```       
sample = {'label1': label1, 'label2': label2, 'image1': image1,
                    'image2': image2, 'rt': rt, 'acc': acc} 
```
where the labels are the labels of the two respective images, images are torch tensor representations of the images, rt is the associated psychophysical reaction time with the images, and sigma is the blurring parameter used for the standard sklearn Gaussian blur. The method also has some commented out parts where you can mess around with blurring one of the images.


## Usage

### 1. Install Depedencies

```
python3 venv -m env
source env/bin/activate
pip3 install -r requirements.txt
```

### 2. Get files

You will need `.csv` files, directories, and pre-trained models from this [link](https://drive.google.com/drive/folders/1mCEpZP8rmN-4SvF1QQVH5qWSQC7LtUv_?usp=sharing)
in order to run the larger experiments.

### 3. Run via scripts

The following scripts perform different deep learning tasks. If working on a server, the comments on the top headers of the script can give gpu commands.

`train.sh` trains the neural network under the same random seeds as in the paper. It defaults to reaction time as the psychophysical parameter, but you can switch to accuracy or no parameter at all.

`eval.sh` tests the saved model from training for quick evaluation of the network. Pre-trained models are also provided on the google drive. Load one into the repo directory and have fun. 

`tune.sh` utilizes the hyperparameter tuning used early on in the experiments. You can run this script and make adjustments according to the [optuna](https://optuna.org/) docs. The script now utilizes just the hyperparameter tuning, but you could use it to tune the model if you would like to. We decided not to focus on the simplicity of the effectiveness of psychophysical parameterization of the label space.  

First, you made me to set your path like: 
```
echo $BASE_PATH
```
and format this directory and subdirectories before the working directory that you cloned for this repo. 

Then, you can call any script like: 
```
bash scripts/train.sh
```

(you could call it with `zsh`, too. and you can modify the script files
with respect to arguments in `argparser`.)

### 4. Run manually

If you are running locally, you may also execute this command (or something similar), and change any of the CLI arguments: 

```
python "$BASE_PATH/main.py" --num_epochs=20  --dataset_file="small_dataset.csv"
```

### 5. Utilize Neptune

This experiment uses Neptune for logging. To utilize this feature, set the flag `use-neptune` to 
`True` and follow their [instructions](https://neptune.ai/) to export your custom API key. 