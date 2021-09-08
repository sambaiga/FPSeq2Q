import torch
import math
import torch.nn as nn
from net.layers import create_linear
import torch.nn.functional as F
import torch.distributions as dist
from pyro.ops.stats import quantile
from .metrics import cwi_score
from pyro.contrib.forecast import eval_crps
import numpy as np
#q = [0.005, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.965, 0.975]
q=(0.1, 0.5, 0.9)
#q = np.linspace(0, 1, 84)[1:-1].tolist()



def get_95_quantile_prediction(gmm=None, n_sample=1000, samples=None, q=q):
    if gmm is not None and samples is None:
        samples = gmm.sample(sample_shape=(n_sample,))
   
    quantiles = quantile(samples, q).permute(1,0,2)
    _, p50,_ = quantile(samples, (0.1, 0.5, 0.9)).squeeze(-1)
    return p50, quantiles, samples


class MDGMM(nn.Module):
    def __init__(self, in_dims=1,out_dims=1,
                 kmix=5,activation=nn.ReLU(),   min_std = 0.01):
        super().__init__()
        self.activation = activation
        self.in_dims = in_dims
        self.out_dim = out_dims
        self.kmix = kmix
        self.min_std = min_std
       
        self._pi = create_linear(self.in_dims,self.kmix)
        self._mu = create_linear(self.in_dims,self.kmix*self.out_dim)
        nn.init.uniform_(self._mu.bias, a=-2.0, b=2.0)
        self._sigma = create_linear(self.in_dims,self.kmix*self.out_dim)
        
                                     
    def forward(self, x):
        
        pi = torch.softmax(self._pi(x), -1)
        mu = self._mu(x).reshape(-1,self.kmix, self.out_dim)
        
        log_var = self._sigma(x).reshape(-1,self.kmix, self.out_dim)
        log_var = F.logsigmoid(log_var)
        log_var = torch.clamp(log_var, math.log(self.min_std), -math.log(self.min_std))
        sigma = torch.exp(0.5 * log_var)
       
        
        mix = dist.Categorical(pi)
        comp = dist.Independent(dist.Normal(mu, sigma), 1)
        gmm = dist.MixtureSameFamily(mix, comp)

        return pi, mu, sigma, gmm
    
    def log_nlloss(self, y, gmm):
        logprobs = gmm.log_prob(y)
        loss = -torch.mean(logprobs)
        return loss    
    
    def sample(self, gmm, n_sample=1000, q=(0.1, 0.5, 0.9)):
        samples = gmm.sample(sample_shape=(n_sample,))
        p10, p50, p90 = quantile(samples, q).squeeze(-1)
        return p10, p50, p90, samples
    




class MDNBlock(MDGMM):
    def __init__(self,  in_dims=1,out_dims=1,
                 kmix=5,activation=nn.ReLU(),   
                 min_std = 0.01, 
                 dist_type="normal", 
                 latent_size=128, 
                 quantile=(0.1, 0.5, 0.9),
                 soft_max_type='softmax',
                 alpha=0.5,
                 kappa=0.5
                ):
        super().__init__(in_dims=in_dims,
                        out_dims=out_dims,
                        kmix=kmix,
                        activation=activation,  
                        min_std = min_std, 
                        dist_type=dist_type,
                        soft_max_type=soft_max_type)
        
        self.feats = create_linear(in_dims,latent_size)
        self.mdn   = MDGMM(in_dims=latent_size,
                            out_dims=out_dims,
                            kmix=kmix,
                            activation=activation,   
                            min_std = min_std, 
                            dist_type=dist_type,
                            soft_max_type=soft_max_type)
        self.q = torch.tensor(quantile)
        self.alpha=alpha
       
        
    def forward(self, x):
        feats = torch.nn.functional.silu(self.feats(x))
        pi, mu, sigma, gmm,  entropies= self.mdn(feats)
        return  pi, mu, sigma, gmm,  entropies


    def step(self, x, y, gmm, entropies):
        
        loss_nllos = self.mdn.log_nlloss(y, gmm)
        p50, samples = self.mdn.sample(gmm,n_sample=1000)
        loss = loss_nllos*(1-self.alpha) + torch.nn.HuberLoss(delta = 0.5)(p50, y)*self.alpha + (-torch.mean(entropies))*self.alpha
        p50, q_samples, samples = get_95_quantile_prediction(gmm, 1000, samples)
        lower = q_samples[:, 0,:]
        upper = q_samples[:, -1,:]
        
        cwi= cwi_score(y, gmm.sample(), lower, upper, eps=1e-6, conf=0.95)
        crps = eval_crps(samples, y)
        mae  = (y-p50).abs().mean()
        return loss, mae, cwi, crps

    
        
        

           
        