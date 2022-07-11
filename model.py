from torch import nn
from torchvision.models import resnet50, vgg16
from transformers import ViTForImageClassification

class LayerAdjustedGenericModel(nn.Module): 
    def __init__(self, model_name, num_classes, device):
        super(LayerAdjustedGenericModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes

        if self.model_name =='resnet':
            self.model = resnet50(pretrained=True).to(device)
            # self.fc = nn.Linear(2048, self.num_classes).to(device)
        elif self.model_name=='vgg':
            self.model = vgg16(pretrained=True).to(device)
            # self.fc = nn.Linear(1000, self.num_classes).to(device)
        elif self.model_name == 'vit':
            self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',num_labels=self.num_classes).to(device)
            # self.fc = nn.Linear(4096, self.num_classes).to(device)
        else: 
            print('model not clearly defined')

        self.model.train()
        # self.fc.train()

    def forward(self, x):
        x = self.model(x)
        # x = self.fc(x) 
        return x 
