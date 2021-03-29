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

        self.num_channels_encoder = np.array([64, 64, 128, 256, 512])
        
        # upconv

        # iconv

        # disp

    def forward(self, x): 
        pass


if __name__ == '__main__':
    input_size = (3, 64, 64)
    device = torch.device("cpu")
    image = np.random.rand(1, 3, 64, 64)
    image = torch.from_numpy(image)
    encoder = Featencoder(50)
    feats_encoded =  encoder.forward(image.float())
    print(feats_encoded)
    decoder = FeatDecoder()
    feats_decoded = decoder(feats_encoded)
