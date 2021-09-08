import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from .layers import create_linear


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear, activation=nn.GELU()):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        activation,
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


class MLPMixerEncoder(nn.Module):
    def __init__(self, in_size=1,
                    dim=128,
                    patch_size=2, 
                    context_size=96,
                    expansion_factor = 4,
                    dropout = 1.0,
                    depth=4,
                    activation=nn.GELU()
                     ):
        
        super().__init__()
        
        assert (context_size % patch_size) == 0, 'sequence must be divisible by patch size'
        num_patches = context_size//patch_size


        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        self.net = nn.Sequential(Rearrange('b c (h p1) -> b h (p1 c)', p1 = patch_size),
                    nn.Linear(patch_size * in_size, dim), *[nn.Sequential(
                    PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first, activation)),
                    PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last, activation))) for _ in range(depth)], 
                    nn.LayerNorm(dim))
        
    def forward(self, x):
        return self.net(x.permute(0,2,1))




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
        layers.append(nn.Sequential(create_linear(feats, latent_dim*output_size), activation))
        self.mlp_network =  nn.ModuleList(layers)
        
    def forward(self, x):
        x = x.flatten(1,2)
        for m in self.mlp_network:
            x = m(x)
        return x.reshape(x.size(0),self.output_size, -1)



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
        for name, param in  self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                if param.ndim>1:
                    nn.init.kaiming_uniform_(param)

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
        
        if self.hidden.size(1)!=input.size(0):
            self.hidden = self.init_hidden_state(input).to(input.device)
        output, hidden = self.gru(input, self.hidden)
        output = self.activation(output)
        self.hidden = hidden.data
        return output



class LSTMEncoder(nn.Module):
    def __init__(self, in_size=1, hidden_size=64, num_layers=2, bidirectional=False, dropout=0.1, activation=nn.GELU()):
        super().__init__()
        self.hidden_size = hidden_size
        self.activation = activation
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden = None
        self.cell = None
        self.lstm = nn.LSTM(in_size, hidden_size, num_layers, 
                          dropout=dropout if num_layers>1 else 0, 
                          batch_first=True, 
                          bidirectional=bidirectional)
        self.nom = nn.LayerNorm(hidden_size)
        self.bidirectional = bidirectional


        for name, param in  self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                if param.ndim>1:
                    nn.init.kaiming_uniform_(param)

       
      
       
    
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
        if self.hidden is None:
            self.hidden, self.cell = self.init_hidden_state(input)
            self.hidden = self.hidden.to(input.device)
            self.cell = self.cell.to(input.device)
        else:
            self.hidden = self.hidden.to(input.device)
            self.cell = self.cell.to(input.device)
        
       
        
        output, (self.hidden, self.cell) = self.lstm(input, (self.hidden, self.cell))
        output, _ = self.lstm(input)
        output = self.activation(output)
        #self.hidden, self.cell = self.hidden.data, self.cell.data
        return output

