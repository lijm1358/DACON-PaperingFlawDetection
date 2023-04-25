import torch
from torch import nn
import timm


class EfficientNetB0(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=19)
    
    def forward(self, x):
        return self.model(x)