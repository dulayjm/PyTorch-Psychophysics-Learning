import argparse
import time

import neptune
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision.models import resnet50

from dataset import OmniglotReactionTimeDataset
from psychloss import RtPsychCrossEntropyLoss
from psychloss import AccPsychCrossEntropyLoss

# args
parser = argparse.ArgumentParser(description='Training Psych Loss.')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='number of epochs to use')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--num_classes', type=int, default=100,
                    help='number of classes')
parser.add_argument('--learning_rate', type=float, default=0.0001, 
                    help='learning rate')
parser.add_argument('--loss_fn', type=str, default='psych-rt',
                    help='loss function to use. select: cross-entropy, psych-rt, psych-acc')                 
parser.add_argument('--dataset_file', type=str, default='small_dataset.csv',
                    help='dataset file to use. out.csv is the full set')
parser.add_argument('--use_neptune', type=bool, default=False,
                    help='log metrics via neptune')

args = parser.parse_args()

if args.use_neptune:
    # choose within your local path setup
    # eg. neptune_path = 'alice/psyphy-loss' ...
    neptune_path = ''
    if neptune_path:
        neptune.init(neptune_path)
        neptune.create_experiment(name='sandbox-{}'.format(args.loss_fn), params={'lr': args.learning_rate}, tags=[args.loss_fn])
    else: 
        print('Please enter a correct neptune path aligned with an existing neptune project.')

# seed for test replication
random_seed = 5 ** 3
torch.manual_seed(random_seed)

# configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device is', device)

num_epochs = args.num_epochs
batch_size = args.batch_size

model = resnet50(pretrained=True).to(device)
model.fc = nn.Linear(2048, args.num_classes).to(device)

# LOAD PATH
# should be just in the project directory
# you can change to whatever you prfer
load_path = 'rt-mod.pth'

model.load_state_dict(torch.load(load_path))

optim = torch.optim.Adam(model.parameters(), 0.001)

if args.loss_fn == 'cross-entropy':
    loss_fn = nn.CrossEntropyLoss()

# transforms and data loader
train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ])
dataset = OmniglotReactionTimeDataset(args.dataset_file, 
            transforms=train_transform)

validation_split = .2
shuffle_dataset = True

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# NOTE: not using train_loader at all 
_ = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

model.eval()

accuracies = []
losses = []

exp_time = time.time()

correct = 0.0
total = 0.0

with torch.no_grad():

    for idx, sample in enumerate(validation_loader):
        image1 = sample['image1']
        image2 = sample['image2']

        label1 = sample['label1']
        label2 = sample['label2']

        if args.loss_fn == 'psych-acc':
            psych = sample['acc']
        else: 
            psych = sample['rt']

        # concatenate the batched images for now
        inputs = torch.cat([image1, image2], dim=0).to(device)
        labels = torch.cat([label1, label2], dim=0).to(device)

        psych_tensor = torch.zeros(len(labels))
        j = 0 
        for i in range(len(psych_tensor)):
            if i % 2 == 0: 
                psych_tensor[i] = psych[j]
                j += 1
            else: 
                psych_tensor[i] = psych_tensor[i-1]
        psych_tensor = psych_tensor.to(device)
        outputs = model(inputs).to(device)

        # we don't update weights at test time
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f'accuracy: {accuracy:.2f}%')

    if args.use_neptune: 
        neptune.log_metric('test accuracy', accuracy)

    accuracies.append(accuracy)

    print(f'{time.time() - exp_time:.2f} seconds')