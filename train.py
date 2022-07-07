import argparse
import time

import numpy as np
from numpy.random.mtrand import seed
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision.models import resnet50, vgg16
from transformers import ViTForImageClassification

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
parser.add_argument('--model_name', type=str, default='resnet',
                    help='number of classes')
parser.add_argument('--learning_rate', type=float, default=0.001, 
                    help='learning rate')
parser.add_argument('--loss_fn', type=str, default='psych-rt',
                    help='loss function to use. select: cross-entropy, psych-rt, psych-acc')                 
parser.add_argument('--dataset_file', type=str, default='small_dataset.csv',
                    help='dataset file to use. out.csv is the full set')
parser.add_argument('--use_neptune', type=bool, default=False,
                    help='log metrics via neptune')

args = parser.parse_args()

# 5 iterations, changing the random seed each time
for seed_idx in range(1, 2):
    random_seed = seed_idx ** 3
    torch.manual_seed(random_seed)

    

    # configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device is', device)

    num_epochs = args.num_epochs
    batch_size = args.batch_size


    if args.model_name=='resnet':
        model = resnet50(pretrained=True).to(device)
        model.fc = nn.Linear(2048, args.num_classes).to(device)
        model.train()
    elif args.model_name=='vgg':
        model = vgg16(pretrained=True).to(device)
        model.fc = nn.Linear(4096, args.num_classes).to(device)
        model.train()
    elif args.model_name == 'vit':
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=100)
        model.fc = nn.Linear(4096, args.num_classes).to(device)
        model.train()

    optim = torch.optim.Adam(model.parameters(), args.learning_rate)

    if args.loss_fn == 'cross-entropy':
        loss_fn = nn.CrossEntropyLoss()

    # data transforms and loader 
    train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    ])
    dataset = OmniglotReactionTimeDataset(args.dataset_file, 
                transforms=train_transform)

    test_split = .2
    shuffle_dataset = True

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    # test loader not utilzed in the train file 
    _ = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)


    accuracies = []
    losses = []
    exp_time = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0.0
        total = 0.0

        for idx, sample in enumerate(train_loader):
            image1 = sample['image1']
            image2 = sample['image2']

            label1 = sample['label1']
            label2 = sample['label2']

            if args.loss_fn == 'psych-acc':
                psych = sample['acc']
            else: 
                psych = sample['rt']

            # concatenate the batched images
            inputs = torch.cat([image1, image2], dim=0).to(device)
            labels = torch.cat([label1, label2], dim=0).to(device)

            # apply psychophysical annotations to correct images
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

            if args.loss_fn == 'cross-entropy':
                loss = loss_fn(outputs, labels)
            elif args.loss_fn == 'psych-acc': 
                loss = AccPsychCrossEntropyLoss(outputs, labels, psych_tensor).to(device)
            else:
                loss = RtPsychCrossEntropyLoss(outputs, labels, psych_tensor).to(device)

            # update weights and back propogate
            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.item()

            # calculate accuracy per class
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total

        print(f'epoch {epoch} accuracy: {accuracy:.2f}%')
        print(f'running loss: {train_loss:.4f}')

        if args.use_neptune: 
            neptune.log_metric('train_loss', train_loss)
            neptune.log_metric('accuracy', accuracy)

        accuracies.append(accuracy)
        losses.append(train_loss)

    print(f'{time.time() - exp_time:.2f} seconds')

    # TO SAVE MODEL: 
    # model will save based upon your loss function
    # set path here :
    path = ''
    if path: 
        torch.save(model.state_dict(), path)