import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
import numpy as np
import pytorch_lightning as pl


class model_pil(pl.LightningModule):
    
    def __init__(self, net, hparams, lr=2e-4, optimizer_name = "Adam", beta1=0.98, momentum=0.9):
        super().__init__()
        self.model = net
        self.save_hyperparameters()
        self.hparams.update(hparams)
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        
        mse_loss, q_loss,  tau_loss, mae,  entropy_loss, cwi = self.model.step(batch)
        self.log("train_mse_loss", mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_entropy_loss', entropy_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_cwi_score', cwi, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_qloss', q_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_tau_loss', tau_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        if optimizer_idx == 0:
            loss = q_loss+mse_loss
        
            return  loss

        if optimizer_idx == 1:
            loss = tau_loss+entropy_loss 
            return  loss
    
    
    def validation_step(self, batch, batch_idx):
        
        mse_loss, q_loss,  tau_loss, mae,  entropy_loss, cwi = self.model.step(batch)
        loss =  q_loss + tau_loss + entropy_loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mse_loss", mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', mae, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_entropy_loss',  entropy_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_cwi',  cwi, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_qloss', q_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_tau_loss', tau_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
    def test_step(self, batch, batch_idx,net):    
        
        taus, tau_hats, q, quantile_hats = net.step(batch, test=True)
    
        
        
        logs = {"pred": q, "tau": taus, "tau_hat":tau_hats, "q_pred":quantile_hats, "true":batch[-1], "cont_feats":batch[0], "cat_feats":batch[1]}
        
        return logs

    def test_epoch_end(self, outputs):
        
        pred = torch.cat([x['pred'] for x in outputs], 0).detach()  
        cat = torch.cat([x['cat_feats'] for x in outputs], 0).detach() 
        true = torch.cat([x['true'] for x in outputs], 0).detach() 
        cont = torch.cat([x['cont_feats'] for x in outputs], 0).detach() 
        tau = torch.cat([x['tau'] for x in outputs], 0).detach() 
        tau_hat = torch.cat([x['tau_hat'] for x in outputs], 0).detach() 
        q_pred = torch.cat([x['q_pred'] for x in outputs], 0).detach() 
       
        results = {"pred": pred, "tau": tau, "tau_hat":tau_hat, "q_pred":q_pred,  "true":true, "cont_feats":cont, "cat_feats":cat}


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

        if self.hparams.optimizer_name == "Adam":
        
            optim = torch.optim.Adam(list(self.model.encoder.parameters())+
            list(self.model.horizon.parameters())+
            list(self.model.quantile_net.parameters())+
            list(self.model.multihead_attention.parameters())+
            list(self.model.tau_cosine.parameters()), lr=self.hparams.lr, eps=1e-2/self.hparams.batch_size)

        elif self.hparams.optimizer_name == "AdamW":
           optim = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,weight_decay=0)

        elif self.hparams.optimizer_name == "SGD":
           optim = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum)

        #fraction_optim = torch.optim.RMSprop(self.model.tau_proposal.parameters(), lr=self.hparams.lr*0.1, alpha=0.95, eps=0.00001)
        fraction_optim = torch.optim.Adam(self.model.tau_proposal.parameters(), lr=self.hparams.lr*0.1,  eps=1e-4)

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 
                                                           patience=50, factor=0.1,
                                                           verbose=True, mode="min")
       
        scheduler = {'scheduler':sched, 
                 'monitor': 'val_mae',
                'interval': 'epoch',
                'frequency': 1}
       
        
        return [optim, fraction_optim], [scheduler]