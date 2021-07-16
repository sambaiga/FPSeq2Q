import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import create_linear, TimeDEmbedding

def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

 #https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/11d70bb428e449fe5384654c05e4ab2c3bbdd4cd/fqf_iqn_qrdqn/network.py#L54
class NQuantileProposal(nn.Module):
    def __init__(self, N=10, z_dim=64, M=48, eps=1e-15, tau=10):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(z_dim, N*M)).apply(lambda x: initialize_weights_xavier(x, gain=0.01))

        create_linear(z_dim, N*M)
        self.N = N
        self.M = M
        self.z_dim = z_dim
        self.eps = eps
        self.tau=tau

    def forward(self, z):
        batch_size = z.shape[0]
        
        z_out = self.net(z).reshape(batch_size, self.N, self.M)
        # Calculate probabilities  and log probabilities using gumble softamx.
        #probs     = F.gumbel_softmax(z_out, tau=1, dim=1)
        #log_probs = torch.log(probs  + self.eps)
        # Calculate (log of) probabilities q_i in the paper.
        log_probs = F.log_softmax(z_out, dim=1)
        probs = log_probs.exp()
        assert probs.shape == (batch_size, self.N, self.M)
        
        tau_0 = torch.zeros(batch_size, 1, self.M, dtype=z.dtype, device=z.device)
        taus_1_N = torch.cumsum(probs, dim=1)

        # Calculate \tau_i (i=0,...,N).
        taus = torch.cat((tau_0, taus_1_N), dim=1)
        
        assert taus.shape == (batch_size, self.N+1, self.M)
        
        # Calculate \hat \tau_i (i=0,...,N-1).
        tau_hats = (taus[:, :-1, :] + taus[:, 1:, :]).detach() / 2.
        assert tau_hats.shape == (batch_size, self.N, self.M)
        
        # Calculate entropies of value distributions.
        entropies = torch.sum(-log_probs*probs, 1)
        assert entropies.shape == (batch_size, self.M)

        return taus, tau_hats, entropies




class NCosineTau(nn.Module):

    def __init__(self, num_cosines=32, z_dim=128, M=48, activation=nn.LeakyReLU()):
        super().__init__()
        self.net = nn.Sequential(TimeDEmbedding(num_cosines*M, z_dim),activation)
        self.num_cosines = num_cosines
        self.z_dim = z_dim
        self.M = M


    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines+1, dtype=taus.dtype,
            device=taus.device).view(1, 1, 1, self.num_cosines)

        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(
            taus.view(batch_size, N, -1, 1) * i_pi
            )

        # Calculate embeddings of taus.
        tau_z = self.net(cosines.flatten(2,3))
        return tau_z