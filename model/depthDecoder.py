#!/home/ash/anaconda3/envs/pytorch/bin/python
import numpy as np
import torch
import torch.nn as nn

from torchsummary import summary
from torch.autograd import Variable
import torch.nn.functional as F

from layers import conv1x1

class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'pointwise'), conv1x1(in_planes if (i == 0) else out_planes, out_planes, False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'pointwise'))(top)
            x = top + x
        return x

class CRN(nn.Module):
    '''
    Cascaded refinement network
    '''
    def __init__(self, in_planes, out_planes, stages):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(conv1x1(in_planes, out_planes, stride=1))
        for i in range(1, stages):
            self.layers.append(conv1x1(out_planes, out_planes, stride=1))
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.stages):
            top = self.maxpool(top)
            top = layers[i](x)
            x = top + x
        return x



class DepthDecoder(nn.Module):
    '''
    The decoder for depth is implemented in a cascaded refinement manner, 
    which decodes depth maps in a top-down pathway. Specifically, multiple-
    scale features from encoder are used to predict maps of corresponding
    sizes via a 3 × 3 convolution followed by sigmoid, and these maps are 
    refined in a coarse-to-fine manner towards the final depth map. Both 
    FeatureNet and DepthNet take image size of 320 × 1024 as inputs.
    '''
    def __init__(self, num_ch_enc):
        super().__init__()
        self.num_ch_enc = num_ch_enc

