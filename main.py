import argparse
import time

import neptune
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50

from dataset import OmniglotReactionTimeDataset
from psychloss import PsychCrossEntropyLoss

# args
parser = argparse.ArgumentParser(description='Training Psych Loss.')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='number of epochs to use')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--num_classes', type=int, default=100,
                    help='number of classes')
parser.add_argument('--learning_rate', type=int, default=0.001,
                    help='learning rate')
parser.add_argument('--loss_fn', type=str, default='psych-rt',
                    help='loss function to use. select: cross-entropy, psych-rt, psych-acc')                 
parser.add_argument('--dataset_file', type=str, default='processed.csv',
                    help='dataset file to use. out.csv is the full set')
parser.add_argument('--use_neptune', type=bool, default=False,
                    help='log metrics via neptune')

args = parser.parse_args()

if args.use_neptune:
    neptune.init('dulayjm/psyphy-loss')
    neptune.create_experiment(name='sandbox-00', params={'lr': args.learning_rate}, tags=['resnet50', 'psych'])

# configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device is', device)

model = resnet50(pretrained=True)
# model.fc = nn.Flatten()
model.fc = nn.Linear(2048, 100)
model = nn.DataParallel(model).to(device)

optim = torch.optim.SGD(model.parameters(), 0.001,
                                 momentum=0.9,
                                weight_decay=0.9)

if args.loss_fn == 'cross-entropy':
    loss_fn = nn.CrossEntropyLoss()

num_epochs = args.num_epochs

batch_size = args.batch_size

train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ])
train_set = OmniglotReactionTimeDataset(args.dataset_file, 
            transforms=train_transform)

dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True)

model.train()

accuracies = []
losses = []

exp_time = time.time()

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0.0

    for idx, sample in enumerate(dataloader):
        image1 = sample['image1']
        image2 = sample['image2']

        label1 = sample['label1']
        label2 = sample['label2']

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

        if args.loss_fn == 'cross-entropy':
            loss = loss_fn(outputs, labels)
        else: 
            loss = PsychCrossEntropyLoss(outputs, labels, psych_tensor).to(device)

        optim.zero_grad()
        loss.backward()
        optim.step()

        running_loss += loss.item()
        labels_hat = torch.argmax(outputs, dim=1)
        correct += torch.sum(labels.data == labels_hat).item()

    train_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / len(train_set)
    print(f'epoch {epoch} accuracy: {accuracy:.2f}%')
    print(f'running loss: {train_loss:.4f}')

    if args.use_neptune: 
        neptune.log_metric('train_loss', train_loss)
        neptune.log_metric('accuracy', accuracy)

    accuracies.append(accuracy)
    losses.append(train_loss)

print(f'{time.time() - exp_time:.2f} seconds')

# plt, do metrics with
