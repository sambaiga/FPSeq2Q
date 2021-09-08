

from net.layers import create_linear
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .RNNBaseline import RNNBaseline
from .mdn_block import MDGMM, get_95_quantile_prediction
import pytorch_lightning as pl
from pyro.contrib.forecast import eval_crps
from .metrics import cwi_score

from tqdm import tqdm
import sys


class RNNMDNModel(RNNBaseline):
    def __init__(self, input_size=1, hidden_size=64, 
                num_layers=2, latent_size=128, 
                 bidirectional=False, dropout=0.1, 
                 activation=nn.LeakyReLU(), 
                 out_size=48, context_size=144,
                 cell_type='GRU',
                 kmix=5,   
                 min_std = 0.01, 
                 dist_type="normal", 
                 quantile=(0.1, 0.5, 0.9),
                 soft_max_type='gumble',
                 alpha=0.8,
                 ):
        super().__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                 bidirectional=bidirectional, dropout=dropout, activation=activation, out_size=out_size, 
                         context_size=context_size, cell_type = cell_type)
        
        self.feats = create_linear(hidden_size*context_size,latent_size)
        self.mdn   = MDGMM(in_dims=latent_size,out_dims=out_size,
                         kmix=kmix,activation=activation,   min_std = min_std)
        self.norm = nn.LayerNorm(hidden_size)
       
        
    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.norm(output)
        feats = torch.nn.functional.silu(self.feats(output.flatten(1,2)))
        pi, mu, sigma, gmm = self.mdn(feats)
        return pi, mu, sigma, gmm
    
    
    def step(self, batch, test=False):
        
        x, y = batch
        pi, mu, sigma, gmm = self(x)
        
        if test:
            return pi, mu, sigma, gmm
        
        else:
            loss_nllos = self.mdn.log_nlloss(y, gmm)
            p10, p50, p90, samples = self.mdn.sample(gmm)
            mse = F.mse_loss(p50, y)
            
            mae  = (y-p50).abs().mean()
            loss = loss_nllos*0.5 + mse*0.5
            p50, q_samples, samples = get_95_quantile_prediction(gmm, 1000, samples)
            lower = q_samples[:, 0,:]
            upper = q_samples[:, -1,:]
            
            cwi= 0.0
            crps = eval_crps(samples, y)
            return loss, mae, cwi,crps
            





class RNNMDNModel_pil(pl.LightningModule):
    
    def __init__(self, net, hparams, lr=2e-4):
        super().__init__()
        self.model = net
        self.save_hyperparameters()
        self.hparams.update(hparams)

    def forward(self, x):
        with torch.no_grad():
            pi, mu, sigma, gmm = self.model(x)
            p50,  q_samples, samples =get_95_quantile_prediction(gmm)
        return p50, q_samples, samples
        
        

        
    def training_step(self, batch, batch_idx):
        
        ##get quantile training loss
        loss, mae, cwi,crps = self.model.step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_cwi_score', cwi, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_crsp_score', crps, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        loss, mae, cwi,crps = self.model.step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_cwi',  cwi, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_crps',  crps, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
    
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
           
    def test_step(self, batch, batch_idx,net):    
        
        pi, mu, sigma, gmm = self.model.step(batch, test=True)
        p50,  q_samples, samples =get_95_quantile_prediction(gmm)
        
        logs = {"pred": p50, 'q_pred':q_samples, "true":batch[-1], "feats":batch[0], 'samples':samples, 'q_samples': q_samples}
        
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