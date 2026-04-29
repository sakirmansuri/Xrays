import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2):
    import torchvision.models as models
    import torch.nn as nn

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# def get_model(num_classes=2):

#     model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

#     # Freeze base layers
#     for param in model.parameters():
#         param.requires_grad = False

#     # Replace final layer
#     in_features = model.fc.in_features
#     model.fc = nn.Linear(in_features, num_classes)

#     return model
