from einops.layers.torch import  Reduce, Rearrange
from .layers import  TimeDEmbedding, create_linear
from .loss_functions import calculate_quantile_huber_loss, cwi_score, N_quantile_proposal_loss
from .quantile_block import NQuantileProposal, NCosineTau
from .mlpmixer import MLPMixerEncoder, MLPEncoder, GRUEncoder, LSTMEncoder, FeedForward
from .block import UNETEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
import numpy as np
import pytorch_lightning as pl


class FPQSeq2Q(nn.Module):
       
    def __init__(self,  n_channels: int =1, 
                        out_size=48, 
                        latent_size=128,
                        context_size=96,
                        N=9,
                        dropout=0.1,
                        margin=1e-2, 
                       kappa=0.5,
                       alpha=0.5,
                       patch_size=4, 
                        expansion_factor = 4,        
                        depth=4,
                       num_cosines=64,
                       num_head=8,
                       out_activation=None,
                       huber_loss = True,
                       calibration_loss = True,
                       entropy_loss = True,
                       nrmse_loss = True,
                       nll_loss = True,
                       encoder_type='MLPMixerEncoder',
                       activation=nn.GELU()
                ):
        super().__init__()
        self.margin = margin
        self.kappa = kappa
        self.alpha = alpha
        self.huber_loss = huber_loss
        self.calibration_loss = calibration_loss
        self.entropy_loss = entropy_loss
        self.huber_loss = huber_loss
        self.nrmse_loss = nrmse_loss
        self.nll_loss = nll_loss
        self.out_size = out_size
        self.context_size  = context_size
        self.encoder_type=encoder_type
        self.out_activation=out_activation
        self.activation = activation
        self.patch_size = patch_size
        self.filter_size = 16
        
        if encoder_type == 'MLPMixerEncoder':
            self.encoder = MLPMixerEncoder(in_size=n_channels,
                                         context_size=context_size,
                                         dim=latent_size,
                                        patch_size=patch_size, 
                                        expansion_factor = expansion_factor,
                                        dropout = dropout,
                                        depth=depth
                                         )
            
           
                                         
        elif encoder_type == 'GRUEncoder':
            self.encoder = GRUEncoder(in_size=n_channels, 
                                      hidden_size=latent_size, 
                                      num_layers=depth, 
                                      dropout=dropout, 
                                      activation=activation)
            
          


        elif encoder_type == 'LSTMEncoder':
            self.encoder = LSTMEncoder(in_size=n_channels, 
                                      hidden_size=latent_size, 
                                      num_layers=depth, 
                                      dropout=dropout, 
                                      activation=activation)
            
            
            
        elif encoder_type=="MLPEncoder":
            self.encoder = nn.Sequential(MLPEncoder(in_size=n_channels,
                                         latent_dim=latent_size,
                                         features_start=latent_size//patch_size,
                                          output_size=context_size//patch_size,
                                         activation=activation, 
                                         context_size=context_size
                                         ),activation)
            
            
        elif encoder_type == 'UNETEncoder':
            
            self.encoder = nn.Sequential(UNETEncoder(n_channels=n_channels, 
                                       latent_dim=latent_size,
                                      features_start=latent_size//patch_size, 
                                      num_layers=depth, 
                                      activation=activation),
                                     Rearrange('b n c -> b c n'))
            
            
            
       

       
        self.horizon = MLPMixerEncoder(in_size=n_channels,
                                         context_size=out_size,
                                         dim=latent_size,
                                        patch_size=patch_size//2, 
                                        expansion_factor = expansion_factor,
                                        dropout = dropout,
                                        depth=depth//2,
                                        activation=activation
                                         )
        
        self.num_cosines  = num_cosines
        self.tau_proposal = NQuantileProposal(N=N, z_dim=self.filter_size**2, M=out_size, tau=1)
        self.tau_cosine   = NCosineTau(num_cosines=num_cosines, z_dim=latent_size, M=out_size, activation=activation)
        self.N = N
        self.context_size = context_size
        self.quantile_net = nn.Sequential(FeedForward(self.filter_size*self.filter_size, expansion_factor=1, dropout=dropout/2, activation=self.activation), 
                                          self.activation, nn.Linear(self.filter_size*self.filter_size, out_size))
                                     
        
        
        self.multihead_attention = nn.MultiheadAttention(latent_size, num_head, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
        
    def get_quantile_values(self, tau, h,z, attn_emb):
        B, N, M = tau.size()
        q = self.tau_cosine(tau).permute(1,0,2)
        hz = torch.cat([h, z], 1)
        attn_output_p, attn_output_weights=self.multihead_attention(q,  hz.permute(1,0,2), hz.permute(1,0,2))
        q=torch.mul(q, attn_output_p).permute(1,0,2)
        #q_m_emb = torch.mul(q.unsqueeze(-1), attn_emb.unsqueeze(1))
        q_m_emb = torch.add(q.unsqueeze(-1), attn_emb.unsqueeze(1))
        attn_emb = F.relu(F.adaptive_avg_pool2d(q_m_emb, (self.filter_size, self.filter_size)))
        out = self.quantile_net(attn_emb.flatten(2,3))
        out = out.clip_(min=-1, max=1)
        return out
        
        
    
    def get_quantile_proposals(self, x, z, h):

        #get attention output between z and h
        attn_output, _=self.multihead_attention(h.permute(1,0,2), z.permute(1,0,2), z.permute(1,0,2))
        x_res = x[:,self.out_size:self.context_size,:1]

        #multiply attention output with the net load of the previous day
        res=torch.nn.functional.adaptive_avg_pool1d(x_res.flatten(1,2).unsqueeze(1), h.size(-1))
        res=res.permute(1,0,2).expand_as(attn_output)
        attn_output = F.relu(torch.mul(res.sigmoid(), attn_output))

        #interporate the attention out to be of equl size as the resideual inputs
        attn_output = F.interpolate(attn_output.permute(1, 2, 0),self.out_size)
        res=x_res.permute(0,2,1).expand_as(attn_output)

        #add the residual inputs to attention output
        attn_output = torch.add(attn_output, res)
        
        attn_emb = F.adaptive_avg_pool2d(attn_output, (self.filter_size, self.filter_size))
        taus, tau_hats, entropies =  self.tau_proposal(attn_emb.flatten(1,2))
        return taus, tau_hats, entropies, attn_output 
        
    def forward(self, x):
        
        z = self.encoder(x[:,:self.context_size,:])
        h = self.horizon(x[:,self.context_size:,:])
        
        return z, h


    def step(self, batch, test=False):

        x, y = batch

        z, h=self(x)
        taus, tau_hats, entropies, attn_output = self.get_quantile_proposals(x, z, h)
        quantile_hats = self.get_quantile_values(tau_hats,  h,z, attn_output)
        q = (quantile_hats*(taus[:, 1:, :] - taus[:, :-1,:])).sum(dim=1)

        if test:
            return   taus, tau_hats, q, quantile_hats
        
        else:
        #quantile loss
            y_q    = y.unsqueeze(1).expand_as(quantile_hats)
            error = (y_q - quantile_hats)

            q_loss=calculate_quantile_huber_loss(error, tau_hats.detach(), kappa=self.kappa)

            lower, upper = quantile_hats[:,0, :], quantile_hats[:,-1,:]
            nmpic =  torch.nn.functional.relu(upper-lower).sum(1, keepdims=True)
            true_mpic = 2*y.std()
            penalty = torch.square(torch.nn.functional.relu(true_mpic - nmpic)).mean()
      
            #diff = quantile_hats[:, 1:, :] - quantile_hats[:,:-1,:]
            #penalty = self.kappa * torch.square(torch.nn.functional.relu(self.margin - diff)).mean()
            q_loss = penalty+q_loss
            mse_loss = nn.MSELoss()(q, y) 

            #q_loss
            quantile = self.get_quantile_values(taus[:, 1:-1,:],  h,z,  attn_output)
            tau_loss= N_quantile_proposal_loss(quantile.detach(), quantile_hats.detach(), taus.detach())


            entropy_loss = (- self.margin* entropies).mean()
            loss =  tau_loss + q_loss 
            

            
            cwi=cwi_score(y, quantile_hats)

            mae = error.abs().mean()

            return self.margin*mse_loss, q_loss,  tau_loss, mae,  entropy_loss, cwi   