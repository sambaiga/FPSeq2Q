import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import create_linear

class MultitargetCosine(nn.Module):

    def __init__(self, num_cosines=64, z_dim=128):
        super().__init__()
       
        self.net = nn.Sequential(
            create_linear(num_cosines, z_dim, bn=False),
            nn.ReLU()
        )
        self.num_cosines = num_cosines
        self.z_dim = z_dim

    def forward(self, taus):
        batch_size, C, N = taus.size()
       

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines+1, dtype=taus.dtype,
            device=taus.device).view(1, 1, 1, self.num_cosines)

        cosines = torch.cos(taus.permute(0,2,1)).unsqueeze(-1)*i_pi 
        cosines = cosines.reshape(batch_size*N, C, self.num_cosines)

        # Calculate embeddings of taus.
        tau_z = self.net(cosines).reshape(batch_size, N, C, -1)

        return tau_z


class MultitargetQProposalNet(nn.Module):
    
    def __init__(self, N=10, z_dim=64):
        super().__init__()

        self.net = create_linear(z_dim, N, bn=False)
        self.N = N
        self.z_dim = z_dim

    def forward(self, z):
        batch_size, C, _ = z.size()
        
        # Calculate (log of) probabilities q_i in the paper.
        log_probs = F.log_softmax(self.net(z), dim=-1)
        probs = log_probs.exp()
        
        
        taus_1_N = torch.cumsum(probs, dim=-1)
        tau_0 = torch.zeros((batch_size, C, 1), dtype=z.dtype, device=z.device)
        # Calculate \tau_i (i=0,...,N).
        taus = torch.cat((tau_0, taus_1_N), dim=-1)
        assert taus.shape == (batch_size, C, self.N+1)


        # Calculate \hat \tau_i (i=0,...,N-1).
        tau_hats = (taus[:, :, :-1] + taus[:, :, 1:]).detach() / 2.
        assert tau_hats.shape == (batch_size, C, self.N)
        
         # Calculate entropies of value distributions.
        entropies = -(log_probs * probs).sum(dim=-1, keepdim=True)

    
        return taus, tau_hats, entropies


class QuantileProposalNet(nn.Module):
    #https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/11d70bb428e449fe5384654c05e4ab2c3bbdd4cd/fqf_iqn_qrdqn/network.py#L54
    def __init__(self, N=10, z_dim=64):
        super().__init__()

        self.net = create_linear(z_dim, N)
        self.N = N
        self.z_dim = z_dim

    def forward(self, z):

        batch_size = z.shape[0]

        # Calculate (log of) probabilities q_i in the paper.
        log_probs = F.log_softmax(self.net(z), dim=1)
        probs = log_probs.exp()
        assert probs.shape == (batch_size, self.N)

        tau_0 = torch.zeros((batch_size, 1), dtype=z.dtype, device=z.device)
        taus_1_N = torch.cumsum(probs, dim=1)

        # Calculate \tau_i (i=0,...,N).
        taus = torch.cat((tau_0, taus_1_N), dim=1)
        assert taus.shape == (batch_size, self.N+1)

        # Calculate \hat \tau_i (i=0,...,N-1).
        tau_hats = (taus[:, :-1] + taus[:, 1:]).detach() / 2.
        assert tau_hats.shape == (batch_size, self.N)

        # Calculate entropies of value distributions.
        entropies = -(log_probs * probs).sum(dim=-1, keepdim=True)
        assert entropies.shape == (batch_size, 1)

        return taus, tau_hats, entropies




class CosineTau(nn.Module):

    def __init__(self, num_cosines=64, z_dim=128):
        super().__init__()
       
        self.net = nn.Sequential(
            create_linear(num_cosines, z_dim),
            nn.ReLU()
        )
        self.num_cosines = num_cosines
        self.z_dim = z_dim


    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines+1, dtype=taus.dtype,
            device=taus.device).view(1, 1, self.num_cosines)

        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(
            taus.view(batch_size, N, 1) * i_pi
            ).view(batch_size * N, self.num_cosines)

        # Calculate embeddings of taus.
        tau_z = self.net(cosines).view(
            batch_size, N, self.z_dim)

        return tau_z

