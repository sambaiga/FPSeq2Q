import torch
from tqdm import tqdm
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from net.utils import get_latest_checkpoint
from net.layers import create_linear
from timeit import default_timer
from .mdn_block import MDNBlock, get_95_quantile_prediction
from .metrics import cwi_score
from pyro.contrib.forecast import eval_crps


class RNNGausModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, 
                 bidirectional=False, dropout=0.1, 
                 activation=nn.LeakyReLU(), 
                 out_size=48, context_size=144, cell_type = 'GRU'):
        super().__init__()
        self.hidden_size = hidden_size
        self.activation = activation
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.cell_type = cell_type
        self.bidirectional=bidirectional
        rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}[self.cell_type]
        
        self.rnn = rnn(input_size, hidden_size, num_layers, 
                          dropout=dropout if num_layers>1 else 0, 
                          batch_first=True, 
                          bidirectional=bidirectional)



        # initialize RNN forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.rnn._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)



        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.kaiming_normal_(param.data)
            
        
        self.dropout = nn.Dropout(dropout/ 2)
        self.feats = create_linear(hidden_size*context_size,hidden_size*2)
        
        
        self._mu = nn.Linear(hidden_size*2, out_size)
        self._sigma = nn.Linear(hidden_size*2, out_size)
        self.softplus = nn.Softplus()
        self.hidden, self.cell = None, None

    def init_hidden_state(self, x):
        num_directions = 2 if self.bidirectional else 1
        
        batch_size = x.size(0)
        if self.cell_type=="LSTM":
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
        else:
            
            hidden = torch.zeros( (self.num_layers * num_directions, batch_size, self.hidden_size))
            return hidden
                
        
        
    
    
        
    def forward(self, input):


        if self.cell_type=="LSTM":
            if self.hidden is None:
                self.hidden, self.cell =  self.init_hidden_state(input)
            if  self.hidden.size(1)!=input.size(0):
                self.hidden, self.cell =  self.init_hidden_state(input)
                
            self.hidden = self.hidden.to(input.device)
            self.cell = self.cell.to(input.device)
            output, (self.hidden, self.cell) = self.lstm(input, (self.hidden.detach(), self.cell.detach()))
        else:
            if self.hidden is None:
                self.hidden=self.init_hidden_state(input)
            if  self.hidden.size(1)!=input.size(0):
                self.hidden=self.init_hidden_state(input)
            self.hidden = self.hidden.to(input.device)
            output, self.hidden = self.rnn(input, self.hidden.detach())

        feats = self.dropout(self.feats(output.flatten(1,2)))
             
        feats = torch.nn.functional.silu(feats)
        mu = self._mu(feats)
        sigma = self.softplus(self._sigma(feats))
        
        return mu, sigma
    
    
    def step(self, batch, test=False):
        
        x, y = batch
        mu, sigma = self(x)
        dist = torch.distributions.normal.Normal(mu, sigma)
        
        
        if test:
            return dist, mu, sigma
        else:
            likelihood = dist.log_prob(y)
            loss = -torch.mean(likelihood)
            samples = dist.sample(sample_shape=(1000,))
            sample_mu = torch.median(samples, dim=0)[0]
            sample_sigma = samples.std(dim=0)
            loss = loss + torch.nn.HuberLoss(delta = 0.5)(sample_mu, y)*0.5
            
            p50, q_samples, samples = get_95_quantile_prediction(dist, 1000, samples)
            lower = q_samples[:, 0,:]
            upper = q_samples[:, -1,:]
        
            cwi=0.0
            crps = eval_crps(samples, y)
            mae  = (y-sample_mu).abs().mean()
            return loss, mae, cwi, crps
        
        
        
class RNNGaussModel_pil(pl.LightningModule):
    
    def __init__(self, net, hparams, 
                 lr=1e-3):
        super().__init__()
        self.model = net
        self.save_hyperparameters()
        self.hparams.update(hparams)

    def forward(self, x):
        with torch.no_grad():
            mu, sigma = self.model(x)
            dist = torch.distributions.normal.Normal(mu, sigma)
            p50,  q_samples, samples =get_95_quantile_prediction(dist)
        return p50, q_samples, samples
        
        

        
    def training_step(self, batch, batch_idx):
        
        ##get quantile training loss
        loss, mae, cwi, crps = self.model.step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_cwi_score', cwi, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_crsp_score', crps, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        loss, mae, cwi, crps = self.model.step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_cwi',  cwi, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_crps',  crps, on_step=True, on_epoch=True, prog_bar=False, logger=True)
       
        
        
    def test_step(self, batch, batch_idx,net):    
        x, y = batch
        dist, mu, sigma = self.model.step(batch, test=True)
        samples = dist.sample(sample_shape=(1000,))
        p50, q_samples, samples = get_95_quantile_prediction(dist, 1000, samples)
       
        pred = torch.median(samples, dim=0)[0]
        sample_sigma = samples.std(dim=0)

        logs = {"pred": pred, 'q_pred':q_samples, "true":batch[-1], "feats":batch[0], 'samples':samples, 'q_samples': q_samples,}
        
        return logs

    def test_epoch_end(self, outputs):
        
        q_pred = torch.cat([x['q_pred'] for x in outputs], 0).detach()  
        pred = torch.cat([x['pred'] for x in outputs], 0).detach()  
        feature = torch.cat([x['feats'] for x in outputs], 0).detach() 
        true = torch.cat([x['true'] for x in outputs], 0).detach() 
        samples = torch.cat([x['samples'] for x in outputs], 1).detach()  
        q_samples = torch.cat([x['q_samples'] for x in outputs], 0).detach() 
       
        results = {"pred": pred, 'q_pred':q_pred,
                   "true":true, "feats":feature, 'samples':samples,  'q_samples':q_samples}


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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.0, 0.999))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           patience=50, factor=0.1,
                                                           verbose=True, mode="min")
       
        scheduler = {'scheduler':sched, 
                 'monitor': 'val_mae',
                'interval': 'epoch',
                'frequency': 1}
       
       
        
        return [optimizer], [scheduler]
           