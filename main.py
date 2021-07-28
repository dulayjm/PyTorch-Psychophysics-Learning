import torch
import torch.nn as nn
from torchvision.models import resnet50
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Grayscale

from dataset import OmniglotReactionTimeDataset
# from psychloss import PsychLoss

# CONFIGS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50(pretrained=True).to(device)
# model.fc = nn.Flatten()
# model.fc = nn.Linear(512, 1623)

optim = torch.optim.SGD(model.parameters(), 0.001,
                                momentum=0.9,
                                weight_decay=0.9)

num_epochs = 20

batch_size = 64

criterion = nn.CrossEntropyLoss().to(device)
# criterion = PsychLoss().to(device)

train_transform = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                ])

train_set = OmniglotReactionTimeDataset('small_dataset.csv', 
            transforms=train_transform)

# train_set = torchvision.datasets.Omniglot('/Users/justindulay/research/psychophysics-loss', 
#                                 download=True,
#                                 transform=transforms.Compose([
#                                 transforms.RandomCrop(32, padding=4),
#                                 transforms.Grayscale(num_output_channels=3),
#                                 transforms.RandomHorizontalFlip(),
#                                 transforms.ToTensor(),
#                                 ]))


dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True)

def softmax(x):
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim=1, keepdim=True)

    return exp_x/sum_x

def log_softmax(x):
    return torch.log(softmax(x))

def CrossEntropyLoss(outputs, targets, psych):
    for idx in range(len(psych)):   
        psych[idx] = abs(30000 - psych[idx])

    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]
    outputs = log_softmax(outputs).to(device)
    outputs = outputs[range(batch_size), targets]

    for i in range(len(outputs)):
        outputs[i] += psych[i]

    return - torch.sum(outputs)/num_examples


model.train()

for epoch in range(num_epochs):

    losses = []
    accuracies = []

    for idx, sample in enumerate(dataloader):

        inputs = sample['image1']
        labels = sample['label1']
        psych = sample['rt']

        print('len of psych is ', len(psych))

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        # loss = criterion(outputs, labels)
        loss = CrossEntropyLoss(outputs, labels, psych).to(device)
        print('the shape of the loss ', loss.shape)

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(loss)

        print('finished batch')

print(losses)
