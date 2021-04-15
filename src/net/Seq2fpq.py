import torch
import tqdm
import sys
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
from .loss_functions import calculate_quantile_huber_loss, quantile_proposal_loss, calibration_loss
from .quantile_block import QuantileProposalNet, CosineTau
from pytorch_lightning.metrics import F1
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from net.layers import GLU, create_linear




class TimeDEmbedding(nn.Module):
    def __init__(self,  in_size, out_size):
        super().__init__()
        self.fc = create_linear(in_size, out_size)

    def forward(self, x):
        if len(x.size()) <= 2:
            return x
        else:
            # Squash samples and timesteps into a single axis
            x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
            y = self.fc(x_reshape)

            # We have to reshape Y
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
            return y

class GRUNetwork(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, bidirectional=False, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          dropout=dropout if num_layers>1 else 0, 
                          batch_first=True, 
                          bidirectional=bidirectional)
        self.nom = nn.LayerNorm(hidden_size)
        self.bidirectional = bidirectional

    def encode(self, x):
        _, hidden_state = self.gru(x)
        return hidden_state
    
    def decode(self, x, hidden_state):
        decoder_output, hidden_state = self.gru(x, hidden_state)
        return decoder_output, hidden_state
        
    def forward(self, x):
        #hidden_state = self.encode(x)
        #output, _ = self.decode(x, hidden_state)
        output, _ = self.gru(x)
        output = self.nom(output)
        return output

    
class EncoderRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, bidirectional=False, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          dropout=dropout if num_layers>1 else 0, 
                          batch_first=True, 
                          bidirectional=bidirectional)
        self.nom = nn.LayerNorm(hidden_size)
        
    def forward(self, input):
        output, _ = self.gru(input)
        output = self.nom(output)
        return output

class FPQForecast(nn.Module):
       
    def __init__(self,  n_channels: int =1, 
                        out_size=1, 
                        emb_size=32,
                        activation=nn.ELU(),
                        hidden_size=128,
                        latent_size=1024,
                        context_size=96,
                        N=9,
                       margin=1e-2, 
                       num_layers=2
                ):
        super().__init__()
        self.margin = margin
        
        self.out_size = out_size
        self.context_size  = context_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.input_emb  = TimeDEmbedding(n_channels, emb_size)
        self.gru = EncoderRNN(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False, dropout=0.1)
        self.out_emb  = create_linear(context_size*hidden_size, latent_size)
        self.num_cosines = 64
                              
        self.tau_proposal = QuantileProposalNet(N, latent_size)
        self.tau_cosine   = CosineTau(num_cosines=64, z_dim=latent_size)
        self.N = N
    
        self.quantile_net = nn.Sequential(nn.Linear(latent_size, latent_size), nn.ReLU(), 
                                          nn.Linear(latent_size, out_size))
        
        
    def calc_quantile_value(self, emb, tau):
        B = emb.size(0)
        tau_emb = self.tau_cosine(tau)
        embeddings = (emb.view(B, 1, -1) * tau_emb)
        quantile_hats = self.quantile_net(embeddings).view(B, -1, self.out_size)
        return quantile_hats                  
    
    def forward(self, x):
        B = x.size(0)
        in_emb = self.activation(self.input_emb(x))
        z = self.gru(in_emb)
        emb = self.activation(self.out_emb(z.flatten(1,2)))
        return emb

    def step(self, batch, test=False):
        x, y = batch
        emb = self(x)
        taus, tau_hats, entropies =  self.tau_proposal(emb)
        quantile_hats = self.calc_quantile_value(emb, tau_hats)
        q = (quantile_hats*(taus[:, 1:] - taus[:, :-1]).unsqueeze(-1) ).sum(dim=1)
    
        if test:
            return   taus, tau_hats, q, quantile_hats
        
        else:
        #quantile loss
            y_q    = y.unsqueeze(1).expand_as(quantile_hats)
            error = (y_q - quantile_hats)
            q_loss=calculate_quantile_huber_loss(error, tau_hats[..., None].detach(), kappa=0.1)
            
            diff = quantile_hats[:,1:, :] - quantile_hats[:,:-1,:]
            penalty =  calibration_loss(y, quantile_hats)

            #quantile proposal loss
            quantile = self.calc_quantile_value(emb, taus[:, 1:-1]).detach()
            quantile_hats = quantile_hats.detach()
            value_1 = quantile - quantile_hats[:, :-1]
            signs_1 = quantile > torch.cat([quantile_hats[:, :1,:], quantile[:, :-1,:]], dim=1)
            value_2 = quantile - quantile_hats[:, 1:]
            signs_2 = quantile < torch.cat([quantile[:, 1:], quantile_hats[:, -1:]], dim=1)
            gradient_tau = (torch.where(signs_1, value_1, -value_1) + torch.where(signs_2, value_2, -value_2)).view(*value_1.size())
            tau_loss = torch.mul(gradient_tau.detach(), taus[:, 1: -1].unsqueeze(2).expand_as(gradient_tau)).sum(1).mean()
            
            #entropy_loss
            entropy_loss = - (self.margin * entropies).mean()
            mae = (y - q).abs().mean()

            loss = entropy_loss + tau_loss + q_loss +  penalty 
            return loss, q_loss,  tau_loss,mae,  penalty
        

class model_pil(pl.LightningModule):
    
    def __init__(self, net, lr=1e-3):
        super().__init__()
        self.model = net
        self.save_hyperparameters()
        
    def training_step(self, batch, batch_idx):
        
        loss, q_loss, p_loss, mae, entropy = self.model.step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_entropy', entropy, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_qloss', q_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_p_loss', p_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        loss, q_loss, p_loss, mae, entropy = self.model.step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', mae, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_entropy', entropy, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_qloss', q_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_p_loss', p_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
    def test_step(self, batch, batch_idx,net):    
        
        taus, tau_hats, q, quantile_hats = net.step(batch, test=True)
    
        
        
        logs = {"pred": q, "tau": taus, "tau_hat":tau_hats, "q_pred":quantile_hats, "true":batch[1]}
        
        return logs

    def test_epoch_end(self, outputs):
        
        pred = torch.cat([x['pred'] for x in outputs], 0).data.cpu().clamp_(min=0) 
        true = torch.cat([x['true'] for x in outputs], 0).data.cpu().clamp_(min=0)  
        tau = torch.cat([x['tau'] for x in outputs], 0).data.cpu()
        tau_hat = torch.cat([x['tau_hat'] for x in outputs], 0).data.cpu()
        q_pred = torch.cat([x['q_pred'] for x in outputs], 0).data.cpu().clamp_(min=0)
       
        results = {"pred": pred, "tau": tau, "tau_hat":tau_hat, "q_pred":q_pred,  "true":true}


        return results                
        
    def predict(self, model, dataloader):
        outputs = []
        model = model.eval()
        batch_size   = dataloader.batchsize if hasattr(dataloader, 'len') else dataloader.batch_size
        num_batches = len(dataloader)
        values = range(num_batches)
        with tqdm(total=len(values), file=sys.stdout) as pbar:
             with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    logs = self.test_step(batch, batch_idx, model.model)
                    outputs.append(logs)
                    del  batch
                    pbar.set_description('processed: %d' % (1 + batch_idx))
                    pbar.update(1)
                pbar.close()
         
        return self.test_epoch_end(outputs)      
        
        
    
    def configure_optimizers(self):
        
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 
                                                           patience=50, 
                                                           verbose=True, mode="min")
        scheduler = {'scheduler':sched, 
                 'monitor': 'val_mae',
                 'interval': 'epoch',
                 'frequency': 1}
       
        
        return [optim], [scheduler]