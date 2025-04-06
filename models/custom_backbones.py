import torchvision.models as models
import torch.nn as nn


def get_custom_backbone(arch_name):

   if arch_name == 'mobilenet_v2':
       model = models.mobilenet_v2(pretrained=False)
       dim_in = model.classifier[1].in_features
       model.classifier = nn.Identity()
       model.fc = nn.Linear(dim_in, 128)
       return model


   elif arch_name == 'resnet18_tiny':
       model = models.resnet18(pretrained=False)
       model.layer1[0].conv1.out_channels = 32
       model.layer1[0].conv2.out_channels = 32
       dim_in = model.fc.in_features
       model.fc = nn.Linear(dim_in, 128)
       return model


   else:
       raise ValueError(f"Unknown custom backbone: {arch_name}")



