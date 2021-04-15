import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import  ResnetBlock, create_conv1, Up,  Downsample, Swish, Normalize
from .attention import AttnBlock
#https://github.com/CompVis/taming-transformers/blob/master/taming/models/vqgan.py


class Seq2PointBaseline(nn.Module):
    def __init__(self, n_channels=1,
                 latent_dim=32,
                 activation=Swish()):
        super().__init__()
        self.activation = activation
        self.enc_net = nn.Sequential(Downsample(n_channels, 30, kernel_size=3, stride=1, padding=1),
                                    self.activation,
                                    create_conv1(30, 40, kernel_size=3, stride=1, padding=1),
                                    self.activation,
                                    create_conv1(40, 50, kernel_size=3, stride=1, padding=1),
                                    self.activation,
                                    create_conv1(50, latent_dim*2, kernel_size=3, stride=1, padding=1),
                                    nn.Dropout(0.2),
                                    self.activation
                                    )
    
        self.conv_out = create_conv1(latent_dim*2, latent_dim, kernel_size=3, stride=1, padding=1, bn=True)
    
    def forward(self, x):
        if x.ndim!=3:
            x = torch.unsqueeze(x, 1)
        else:
            x = x.permute(0,2,1)
        x = self.enc_net(x)
        out = self.conv_out(x)
        return out



class UNETNILM(nn.Module):
       
    def __init__(
            self, 
            n_channels: int =1, 
            latent_dim: int = 32,
            features_start=16, 
            num_layers=4, 
            activation=Swish()
            ):
        super().__init__()
        self.activation = activation
        layers = [nn.Sequential(Downsample(n_channels, features_start, kernel_size=3, stride=1, padding=1), activation)]
        feats = features_start
        for i in range(num_layers - 1):
            layers.append(nn.Sequential(create_conv1(feats, feats * 2, 3, bias=True, stride=1, padding=1), self.activation))
            feats *= 2
            
        self.enc_layers = nn.ModuleList(layers)
        self.enc_attn = AttnBlock(feats)
        
        layers = []
        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, 3, 1, 1, self.activation))
            feats //= 2 
        
        self.dec_layers = nn.ModuleList(layers)
        self.dec_attn = AttnBlock(feats)
        self.norm = Normalize(feats)
        self.conv_out = create_conv1(feats, latent_dim, kernel_size=3, stride=1, padding=1, bn=False)
        
        
    def forward(self, x):
        if x.ndim!=3:
            x = torch.unsqueeze(x, 1)
        else:
            x = x.permute(0,2,1)
       
        xi = [self.enc_layers[0](x)]
        for layer in self.enc_layers[1:]:
            
            xi.append(layer(xi[-1]))
            
        xi[-1] = self.enc_attn(xi[-1])
        
        for i, layer in enumerate(self.dec_layers):
           
            xi[-1] = layer(xi[-1], xi[-2 - i])
        
        xi[-1] = self.dec_attn(xi[-1])
        out = self.norm(xi[-1])
        out = self.activation(out)
        out = self.conv_out(out)
        return out


class UNETResidual(nn.Module):
   
    def __init__(self, num_layers: int = 4, 
                 features_start: int = 16, 
                 n_channels: int =1, latent_dim=32, 
                 activation=Swish()):
        super().__init__()
        self.num_layers = num_layers
        self.activation = activation
        layers = [nn.Sequential(Downsample(n_channels, features_start, kernel_size=3, stride=1, padding=1), activation)]
        feats = features_start
        for i in range(num_layers - 1):
            layers.append(ResnetBlock(feats, feats * 2))
            feats *= 2
            
        self.enc_attn = AttnBlock(feats)
        self.enc_layers = nn.ModuleList(layers)
        
        layers = []
        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2,  3, 1, 1, self.activation))
            feats //= 2 
        
        self.dec_layers = nn.ModuleList(layers)
        self.dec_attn = AttnBlock(feats)
        self.norm = Normalize(feats)
        self.conv_out = create_conv1(feats, latent_dim, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        if x.ndim!=3:
            x = torch.unsqueeze(x, 1)
        else:
            x = x.permute(0,2,1)
       
        xi = [self.enc_layers[0](x)]
        for layer in self.enc_layers[1:]:
            
            xi.append(layer(xi[-1]))
            
        xi[-1] = self.enc_attn(xi[-1])
            
        for i, layer in enumerate(self.dec_layers):
           
            xi[-1] = layer(xi[-1], xi[-2 - i])
            
        xi[-1] = self.dec_attn(xi[-1])
        out = self.norm(xi[-1])
        out = self.activation(out)
        out = self.conv_out(out)
        return out