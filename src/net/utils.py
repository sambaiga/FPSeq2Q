import numpy as np
import torch
import glob
import os
import pytorch_lightning as pl

def get_latest_checkpoint(checkpoint_path):
    checkpoint_path = str(checkpoint_path)
    list_of_files = glob.glob(checkpoint_path + '/*.ckpt')
    
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
    else:
        latest_file = None
    return latest_file

class DictLogger(pl.loggers.TensorBoardLogger):
    """PyTorch Lightning `dict` logger."""
    # see https://github.com/PyTorchLightning/pytorch-lightning/blob/50881c0b31/pytorch_lightning/logging/base.py

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = [] 

    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step=step)
        self.metrics.append(metrics)



def get_predictions(net, features, true,  experiment):
    with torch.no_grad():
        z, h = net(features)
        taus, tau_hats, entropies, attn_output = net.get_quantile_proposals(features, z, h)
        quantile_hats = net.get_quantile_values(tau_hats,  h,z, attn_output)
        pred = (quantile_hats*(taus[:, 1:, :] - taus[:, :-1,:])).sum(dim=1)
            
            
    pred = experiment.target_transformer.inverse_transform(pred.numpy())
    true = experiment.target_transformer.inverse_transform(true.numpy())
    q_pred = quantile_hats.numpy()
    N, M, T = q_pred.shape
    q_pred = q_pred.reshape(N*M, T)
    q_pred = experiment.target_transformer.inverse_transform(q_pred)
    q_pred = q_pred.reshape(N, M, T)
    
    return true, pred, q_pred, tau_hats.numpy()
