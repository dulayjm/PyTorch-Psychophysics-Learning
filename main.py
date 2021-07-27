import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Grayscale

from dataset import OmniglotReactionTimeDataset

# CONFIGS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50(pretrained=True).to(device)
# model.fc = nn.Flatten()
# model.fc = nn.Linear(512, 1623)

optim = torch.optim.SGD(model.parameters(), 0.001,
                                momentum=0.9,
                                weight_decay=0.9)

num_epochs = 3

batch_size = 64

# TODO: change to psyphy loss
criterion = nn.CrossEntropyLoss().to(device)


# TODO: configure with your dataset, including re-write to new format
# dataset = OmniglotReactionTimeDataset('small_dataset.csv', 
#             transforms=transforms.ToTensor())

train_set = torchvision.datasets.Omniglot('/Users/justindulay/research/psychophysics-loss', 
                                download=True,
                                transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                ]))


dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size, shuffle=True,
        num_workers=batch_size, pin_memory=True)



model.train()

for epoch in range(num_epochs):

    losses = []
    accuracies = []

    for inputs, labels in dataloader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(loss)

        print('finished batch')
