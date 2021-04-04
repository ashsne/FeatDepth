#!/home/ash/anaconda3/envs/pytorch/bin/python
import numpy as np
import torch
import torch.nn as nn

from torchsummary import summary
from torch.autograd import Variable
import torch.nn.functional as F

from resnet import resnet18, resnet34, resnet50, resnet101
from featEncoder import FeatEncoder
from layers import conv3x3, upsample


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = conv3x3(in_channels, out_channels)
        self.nonlin = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.nonlin(x)
        x = upsample(x)
        return x


class FeatDecoder(nn.Module):
    '''
    The decoder contains five 3 × 3 convolutional layers
    and each followed by a bilinear upsampling layer. Multi-scale 
    feature maps from convolutional layers of the decoder are used 
    to generate multi-scale reconstructed images, where feature map
    of each scale further goes through a 3 × 3 convolution with sig-
    moid function for image reconstruction. The largest feature map 
    with 64 channels from encoder is regularized by L dis and L cvt
    and will be used for feature-metric loss.
    '''
    def __init__(self, num_ch_enc, num_output_channels=3):
        super().__init__()

        num_ch_dec = [16, 32, 64, 128, 256]

        # upconv
        self.upconv5 = UpConv(num_ch_enc[4], num_ch_dec[4])
        self.upconv4 = UpConv(num_ch_dec[4], num_ch_dec[3])
        self.upconv3 = UpConv(num_ch_dec[3], num_ch_dec[2])
        self.upconv2 = UpConv(num_ch_dec[2], num_ch_dec[1])
        self.upconv1 = UpConv(num_ch_dec[1], num_ch_dec[0])

        # disp
        self.recon4 = conv3x3(num_ch_dec[3], num_output_channels)
        self.recon3 = conv3x3(num_ch_dec[2], num_output_channels)
        self.recon2 = conv3x3(num_ch_dec[1], num_output_channels)
        self.recon1 = conv3x3(num_ch_dec[0], num_output_channels)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, image_id=0):
        self.outputs = {}
        _, _, _, _,econv5 = input_features
        upconv5 = self.upconv5(econv5)
        upconv4 = self.upconv4(upconv5)
        upconv3 = self.upconv3(upconv4)
        upconv2 = self.upconv2(upconv3)
        upconv1 = self.upconv1(upconv2)
        
        self.outputs[("recon_img", image_id, 3)] = self.sigmoid(self.recon4(upconv4))
        self.outputs[("recon_img", image_id, 2)] = self.sigmoid(self.recon3(upconv3))
        self.outputs[("recon_img", image_id, 1)] = self.sigmoid(self.recon2(upconv2))
        self.outputs[("recon_img", image_id, 0)] = self.sigmoid(self.recon1(upconv1))

        return self.outputs


if __name__ == '__main__':
    device = torch.device("cpu")
    image = np.random.rand(1, 3, 256, 256)
    image = torch.from_numpy(image)
    print("image_size = ", image.size())
    encoder = FeatEncoder(50)
    feats_encoded =  encoder.forward(image.float())
    print("encoded_feats_size = ",feats_encoded[4].size())
    decoder = FeatDecoder(encoder.num_channels_encoder)
    feats_decoded = decoder(feats_encoded)
    print("recon_size = ",feats_decoded[('recon_img', 0, 0)].size())
