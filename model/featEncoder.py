#!/home/ash/anaconda3/envs/pytorch/bin/python

import numpy as np
import torch
import torch.nn as nn
from resnet import resnet18, resnet34, resnet50, resnet101

from torchsummary import summary
from torch.autograd import Variable

class FeatEncoder(nn.Module):
    def __init__(self, num_layers, pretrained_path=None):
        super().__init__()

        resnets = {18: resnet18,
                   34: resnet34,
                   50: resnet50,
                   101: resnet101,}

        if num_layers not in resnets:
            raise ValueError("Valid number of layers are 18, 34, 50 or 101")

        self.encoder = resnets[num_layers]()
        
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            self.encoder.load_state_dict(checkpoint)

        self.num_channels_encoder = np.array([64, 64, 128, 256, 512])
        if num_layers > 34:
            self.num_channels_encoder *= 4

    def forward(self, x): 
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        return x

if __name__ == '__main__':
    input_size = (3, 64, 64)
    device = torch.device("cpu")
    model = FeatEncoder(50)
    image = np.random.rand(1, 3, 64, 64)
    image = torch.from_numpy(image)
    print(model.forward(image.float()))

