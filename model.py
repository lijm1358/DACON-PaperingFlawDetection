import timm
import torch
from torch import nn


class EfficientNetB0(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=19)

    def forward(self, x):
        return self.model(x)
    
class EfficientNetB2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("efficientnet_b2", pretrained=True, num_classes=19)

    def forward(self, x):
        return self.model(x)
    
class EfficientNetB4(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=19)

    def forward(self, x):
        return self.model(x)