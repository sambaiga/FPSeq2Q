import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=4):
    return nn.GroupNorm(num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.mul(x.sigmoid(), x)

def create_linear(in_channels, out_channels, bn=True):
    m = nn.Linear(in_channels,out_channels)
    nn.init.xavier_normal_(m.weight.data)
    if m.bias is not None:
        torch.nn.init.constant_(m.bias, 0)
    if bn:
        bn = nn.BatchNorm1d(out_channels)
        m = nn.Sequential(m, bn)
    return m

class GLU(nn.Module):
    #Gated Linear Unit
    def __init__(self, input_size):
        super(GLU, self).__init__()
        
        self.fc1 = create_linear(input_size,input_size)
        self.fc2 = create_linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)




def create_conv1(in_channels, out_channels, 
                 kernel_size=3, bias=True, 
                 stride=1, padding=0, bn=True):

    
    m = nn.Conv1d(in_channels,out_channels, 
                  kernel_size, 
                  bias=bias, 
                  stride=stride, padding=padding)

    nn.init.xavier_normal_(m.weight.data)
    if m.bias is not None:
        torch.nn.init.constant_(m.bias, 0)

    if bn:
        bn = nn.BatchNorm1d(out_channels)
        m = nn.Sequential(m, bn)
    return m


def create_deconv1(in_channels, out_channels,
                   kernel_size=3, bias=True,
                   stride=1, padding=0, bn=True):

    
    m = nn.ConvTranspose1d(in_channels,
                           out_channels, 
                           kernel_size, 
                           bias=bias, 
                           stride=stride, padding=padding)
    
    nn.init.xavier_normal_(m.weight.data)
    if m.bias is not None:
        torch.nn.init.constant_(m.bias, 0)

    if bn:
        bn = nn.BatchNorm1d(out_channels)
        m = nn.Sequential(m, bn)
    return m

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return create_conv1(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias, bn=False)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return create_conv1(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias, bn=False)


def dilated_conv3x3(in_planes, out_planes, dilation, bias=True):
    m = nn.Conv1d(in_planes, out_planes, kernel_size=3, padding=dilation, dilation=dilation, bias=bias)
    nn.init.xavier_normal_(m.weight.data)
    return m



class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = create_conv1(in_channels, out_channels, 
                 kernel_size=kernel_size, bias=True, 
                 stride=stride, padding=padding, bn=True)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool1d(x, kernel_size=2, stride=2)
        return x



class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_shortcut=False, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = create_conv1(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1, bn=False)
        
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = create_conv1(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1, bn=False)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = create_conv1(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     bn=False)
            else:
                self.nin_shortcut = create_conv1(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    bn=False)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h

class Up(nn.Module):
       
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride=1, padding=1, activation=nn.ReLU()):
        super().__init__()
        
        self.upsample = nn.Sequential(create_deconv1(in_channels=in_ch, 
                                        out_channels=out_ch, 
                                        kernel_size=kernel_size, 
                                        stride=stride,
                                        padding=padding, bn=False),
                                        activation)
        self.conv = nn.Sequential(create_conv1(in_channels=in_ch,
                            out_channels=out_ch, 
                            kernel_size=kernel_size, 
                            stride=stride,padding=padding),
                            activation)
        
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diff = x2.shape[2] - x1.shape[2]
        x1 = F.pad(x1, [diff// 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x  



