#!/home/ash/anaconda3/envs/pytorch/bin/python

import torch
import torch.nn as nn
from torchsummary import summary
from torch.autograd import Variable


def conv3x3(inplanes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(inplanes, out_planes, stride=1):
    # 1x1 convolution
    return nn.Conv2d(inplanes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    # residual block for resnet18 and resnet34
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        # Downsampling
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # residual block for resnet50 and beyond
    # saves exensive computation of 3x3 kernel
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes*self.expansion, stride)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(x)
        out = self.relu(x)
        out = self.conv2(x)
        out = self.bn2(x)
        out = self.relu(x)
        out = self.conv3(x)
        out = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual

        return out






if __name__ == '__main__':
    input_size = (3, 64, 64)
    device = torch.device("cpu")
    block = BasicBlock(3, 10)
    print(block)
    bottleneck_block =  Bottleneck(64, 64).to(device)
    summary(bottleneck_block, input_size)
