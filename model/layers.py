import torch.nn as nn
import torch
import numpy as np

def upsample(x):
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    return x

def conv3x3(inplanes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(inplanes, out_planes, stride=1):
    # 1x1 convolution
    return nn.Conv2d(inplanes, out_planes, kernel_size=1, stride=stride, bias=False)



if __name__ == '__main__':
    input_size = (1, 2, 2)
    device = torch.device("cpu")
    np.random.seed(0)
    torch.manual_seed(0)
    image = np.random.rand(1, 1, 2, 2)
    image = torch.from_numpy(image)
    encoder = conv1x1(1, 1, 1)
    print(encoder.forward(image.float()))
