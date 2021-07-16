import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import  ResnetBlock, create_conv1, Up,  Downsample, Swish, Normalize, create_linear
from .attention import AttnBlock
#https://github.com/CompVis/taming-transformers/blob/master/taming/models/vqgan.py
from einops.layers.torch import  Reduce


    
class GRUEncoder(nn.Module):
    def __init__(self, in_size=1, hidden_size=64, num_layers=2, bidirectional=False, dropout=0.1, activation=nn.GELU()):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(in_size, hidden_size, num_layers, 
                          dropout=dropout if num_layers>1 else 0, 
                          batch_first=True, 
                          bidirectional=bidirectional)
        self.nom = nn.LayerNorm(hidden_size)
        self.activation = activation
        self.hidden = None
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.reduce_mean = Reduce('b n c -> b c', 'mean')

    def init_hidden_state(self, x):
        
        batch_size = x.size(0)
        num_directions = 2 if self.bidirectional else 1
       
        hidden = torch.zeros( (self.num_layers * num_directions, batch_size, self.hidden_size))

        return hidden
        
    def forward(self, input):
        if self.hidden is None:
            self.hidden = self.init_hidden_state(input).to(input.device)
        else:
            self.hidden = self.hidden.to(input.device)
        

        output, hidden = self.gru(input, self.hidden)
        self.hidden = hidden.data
        #output = self.reduce_mean(self.activation(self.nom(output)))
        return output



class LSTMEncoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, bidirectional=False, dropout=0.1, activation=Swish()):
        super().__init__()
        self.hidden_size = hidden_size
        self.activation = activation
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                          dropout=dropout if num_layers>1 else 0, 
                          batch_first=True, 
                          bidirectional=bidirectional)
        self.nom = nn.LayerNorm(hidden_size)
    
    def init_hidden_state(self, x):
        num_directions = 2 if self.bidirectional else 1
        
        batch_size = x.size(0)
        
        hidden = torch.zeros(
            (self.num_layers * num_directions, batch_size, self.hidden_size),
            device=x.device,
            dtype=x.dtype,
        )
        cell = torch.zeros(
            (self.num_layers * num_directions, batch_size, self.hidden_size),
            device=x.device,
            dtype=x.dtype,
        )
        return hidden, cell
        
    def forward(self, input):
        output, _ = self.lstm(input)
        output = self.activation(self.nom(output))
        return output

class ConvEncoder(nn.Module):
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



class UNETEncoder(nn.Module):
       
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
            layers.append(nn.Sequential(create_conv1(feats, feats * 2, 3, bias=False, stride=1, padding=1), self.activation))
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


class UNETResidualEncoder(nn.Module):
   
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


class MLPEncoder(nn.Module):
    def __init__(self, in_size=1, 
                     latent_dim: int = 32,
                    features_start=16, 
                    num_layers=4, 
                 context_size=96,
                 output_size=None, 
                 activation=nn.ReLU(),
                 bn=True):
        
        super().__init__()
        self.in_size = in_size*context_size
        self.context_size = context_size
        self.output_size = output_size
        layers = [nn.Sequential(create_linear(self.in_size , features_start, bn=bn), activation)]
        feats = features_start
        for i in range(num_layers-1):
            layers.append(nn.Sequential(create_linear(feats, feats*2, bn=bn), activation))
            feats = feats*2
        layers.append(nn.Sequential(create_linear(feats, latent_dim), activation))
        self.mlp_network =  nn.ModuleList(layers)
        
    def forward(self, x):
        x = x.flatten(1,2)
        for m in self.mlp_network:
            x = m(x)
        return x