#!/home/ash/anaconda3/envs/pytorch/bin/python

import os

import torch
import torch.nn as nn
from torchsummary import summary
from torch.autograd import Variable
from torch.nn import BatchNorm2d as bn

import numpy as np

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
        
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample=None

        if stride!=1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18(pretrained_path=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(os.path.join(pretrained_path, 'resnet50.pth')))
        print('Loaded pre-trained weights')


def resnet34(pretrained_path=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(os.path.join(pretrained_path, 'resnet50.pth')))
        print('Loaded pre-trained weights')


def resnet50(pretrained_path=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(os.path.join(pretrained_path, 'resnet50.pth')))
        print('Loaded pre-trained weights')
    return model


def resnet101(pretrained_path=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(os.path.join(pretrained_path, 'resnet50.pth')))
        print('Loaded pre-trained weights')


if __name__ == '__main__':
    input_size = (3, 64, 64)
    device = torch.device("cpu")
    model = resnet50()
    image = np.random.rand(1, 3, 64, 64)
    image = torch.from_numpy(image)
    print(model.forward(image.float()))
    print(conv1x1(4, 4))
    print(conv1x1(4, 4, 1))
