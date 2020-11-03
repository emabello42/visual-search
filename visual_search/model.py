import torch
import torch.nn as nn
from torchvision import models, transforms

class ResnetExt(torch.nn.Module):
    def __init__(self):
        super(ResnetExt, self).__init__()
        model = models.resnet50(pretrained = True)

        modules = list(model.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*modules)
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.fc = model.fc
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        out = self.fc(features)
        return out, features