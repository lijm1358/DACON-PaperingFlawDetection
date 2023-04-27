import timm
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
    
class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("resnet50", pretrained=True, num_classes=19)

    def forward(self, x):
        return self.model(x)
    
class ConvNeXtBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('convnext_base_384_in22ft1k', pretrained=True, num_classes=19)
    
    def forward(self, x):
        return self.model(x)
    
class ConvNeXtSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('convnext_small_384_in22ft1k', pretrained=True, num_classes=19)
        
    def forward(self, x):
        return self.model(x)
    
class ConvNeXtTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('convnext_tiny_384_in22ft1k', pretrained=True, num_classes=19)
        
    def forward(self, x):
        return self.model(x)
    
class ViTSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('vit_small_patch32_384', pretrained=True, num_classes=19)

    def forward(self, x):
        return self.model(x)
    
class ViTBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('vit_base_patch32_384', pretrained=True, num_classes=19)

    def forward(self, x):
        return self.model(x)
