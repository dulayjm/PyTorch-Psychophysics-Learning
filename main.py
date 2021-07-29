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
model.fc = nn.Linear(2048, 100)

optim = torch.optim.SGD(model.parameters(), 0.001,
                                momentum=0.9,
                                weight_decay=0.9)

num_epochs = 20

batch_size = 64

criterion = nn.CrossEntropyLoss().to(device)

train_transform = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                ])

train_set = OmniglotReactionTimeDataset('small_dataset.csv', 
            transforms=train_transform)

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

def PsychCrossEntropyLoss(outputs, targets, psych):
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]
    # print('before everything')
    # print(outputs)
    # print('len of outputs', len(outputs))
    # print('shape of outputs', outputs.shape)


    outputs = log_softmax(outputs).to(device)
    # print('after log softmax')
    # print(outputs)
    # print('len of outputs', len(outputs))
    # print('shape of outputs', outputs.shape)

    outputs = outputs[range(batch_size), targets]
    # print('after reshape')
    # print(outputs)

    # print('shape of outputs', outputs.shape)

    # converting reaction time to penalty
    for idx in range(len(psych)):   
        psych[idx] = abs(5000 - psych[idx])

    for i in range(len(outputs)):
        outputs[i] += (psych[i] / 1000)


    # print('after psych added')
    # print('len of outputs', len(outputs))

    # print('len of psych', len(psych))

    return - torch.sum(outputs)/num_examples


model.train()

accuracies = []
losses = []

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

        # print('psychtensor', psych_tensor)

        outputs = model(inputs)

        # print('inputs shape', inputs.shape)
        # print('labels shape', labels.shape)
        # print('outputs shape', outputs.shape)

        # loss = criterion(outputs, labels).to(device)
        loss = PsychCrossEntropyLoss(outputs, labels, psych_tensor).to(device)

        optim.zero_grad()
        loss.backward()
        optim.step()

        running_loss += loss.item()
        
        labels_hat = torch.argmax(outputs, dim=1)
        # train_acc = torch.sum(labels.data == labels_hat).item() / (len(labels) * 1.0)
        correct += torch.sum(labels.data == labels_hat).item()

    train_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / len(train_set)
    print(f'epoch {epoch} accuracy: {accuracy:.2f}%')
    print(f'running loss: {train_loss:.4f}')

    accuracies.append(accuracy)
    losses.append(train_loss)

print('finished training')

# plt, do metrics with
# print(losses)
# print(accuracies)
