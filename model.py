from torch import nn
from torchvision.models import resnet50, vgg16
from transformers import ViTForImageClassification, ViTConfig

class LayerAdjustedGenericModel(nn.Module): 
    def __init__(self, model_name, num_classes, device):
        super(LayerAdjustedGenericModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes

        if self.model_name =='resnet':
            self.model = resnet50(pretrained=True).to(device)
            self.model.fc = nn.Linear(2048, self.num_classes).to(device)
            self.dropout = nn.Dropout(0.25)
        elif self.model_name=='vgg':
            model = vgg16(pretrained=True).to(device)
            self.model = nn.Sequential(*(list(model.children())[:-1]))
            self.fc = nn.Linear(1000, self.num_classes).to(device)
            self.dropout = nn.Dropout(0.25)
            
        elif self.model_name == 'vit':
            #model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            # can pass in a ViTConfig
            configuration = ViTConfig(
                    image_size=224, 
                    patch_size=16,
                    num_labels=100,
                    num_channels=3,
                    hidden_dropout_prob=0.5,
                    return_dict=False, # should make it a torch tensor
                    )
            self.model = ViTForImageClassification.from_pretrained(
                    'google/vit-base-patch16-224-in21k',
                    #num_labels=self.num_classes,
                    config=configuration).to(device)
            # self.fc = nn.Linear(4096, self.num_classes).to(device)
        else: 
            print('model not clearly defined')

        self.model.train()
        # self.fc.train()

    def forward(self, x):
        x = self.model(x)

        if self.model_name == 'vgg':
            x = self.fc(x)
        # x = self.fc(x) 
        x = self.dropout(x)
        return x 
