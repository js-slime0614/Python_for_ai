import torch
from torch.functional import norm
import torch.nn as nn
from torch.nn.modules import padding
from torch.nn.modules.activation import Tanh

class Generator(nn.Module):

    def __init__(self, in_channels, out_channels, norm='bnorm'):
        super(Generator, self).__init__()

        #decoder
        self.dec1 = DECBR2d(1 * in_channels, 8 * 64, kernel_size=4, stride=1, padding=0, norm=norm, relu=0.0, bias=False)

        self.dec2 = DECBR2d(8 * 64, 4 * 64, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, bias=False)

        self.dec3 = DECBR2d(4 * 64, 2 * 64, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, bias=False)

        self.dec4 = DECBR2d(2 * 64, 1 * 64, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, bias=False)

        self.dec5 = DECBR2d(1 * 64, out_channels, kernel_size=4, stride=2, padding=1, norm=None, relu = None, bias=None )

        self.tanh = nn.Tanh()
    
    # x = input image
    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.tanh(x)

        return x

#DCGAN 판별자 만들기
class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, norm='bnorm'):
        super(Discriminator, self).__init__()

        #Encoder
        self.enc1 = CBR2d(in_channels, 1 * 64, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)

        self.enc2 = CBR2d(1 * 64, 2 * 64, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)

        self.enc3 = CBR2d(2 * 64, 4 * 64, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)

        self.enc4 = CBR2d(4 * 64, 8 * 64, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)

        self.enc5 = CBR2d(8 * 64, out_channels, kernel_size=4, stride=2, padding=1, norm=None, relu=None, bias=False)

        self.sig = nn.Sigmoid()

    def forward(self, x):

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        
        x = self.sig(x)    



class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=True, norm='bnorm', relu=0.0):
        
        layers = []
        layers += [nn.Conv2d(in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        padding=padding,
        bias=bias)]

        if norm == 'bnorm':
            layers+=[nn.BatchNorm2d(num_features=out_channels)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    # x : input image
    def forward(self, x):
        return self.cbr(x)

    
class DECBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=True, norm='bnorm', relu=0.0):
        
        layers = []
        layers += [nn.ConvTranspose2d(in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        padding=padding,
        bias=bias)]

        if norm == 'bnorm':
            layers+=[nn.BatchNorm2d(num_features=out_channels)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    # x : input image
    def forward(self, x):
        return self.cbr(x)
