import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from pathlib import Path
import numpy as np
import torch 
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.visual_functions import plot_intervals_ordered, plot_xy, plot_calibration, plot_boxplot_per_encoder, plot_prediction_with_pi, plot_prediction, set_figure_size
from utils.data import DatasetObjective
from experiment.baseline_experiment import q
from utils.metrics import get_daily_metrics, get_daily_pointwise_metrics
from experiment.backtesting import bactesting_sliding, bactesting_expanding
from experiment.baseline_experiment import  fit_baseline, fit_auto_regressive_forecasting, fit_space_state_forecasting, eval_crps, fit_sarimax_forecasting
from utils.optimisation import get_search_params, get_best_params, run_study
from net.model_org import FPQSeq2Q
from net.RNNBaseline import RNNBaseline, RNNBaselineModel_pil
from net.RNNMDNModel import RNNMDNModel, RNNMDNModel_pil
from net.RNNGauss import RNNGausModel, RNNGaussModel_pil
from net.loss_functions import cwi_score
from net.model_pil import model_pil
from net.layers import Swish
from net.utils import DictLogger, get_predictions
from utils.data import TimeSeriesDataset
from  net.utils import get_latest_checkpoint
from IPython.display import set_matplotlib_formats
from timeit import default_timer
set_matplotlib_formats('retina')
az.style.use(["science", "grid"])
warnings.simplefilter("ignore")



#general parameters
activations = [Swish(), nn.ReLU(), nn.ELU(), nn.LeakyReLU(), nn.GELU()]
periods = ('15T', '30T', '60T')
samples = (96, 48, 24)
horizons={2:'hourly', 12:'sixhour', 24:'halfday', 48:'daily'}

class Experiment(object):
    def __init__(self, combine=True,
        add_ghi_feature=False,
        cleanData=True,
        window=slice('2019-03-01', '2020-9-30'),
        train_window=slice('2019-03-01', '2019-09-30'),
        test_window=slice('2019-10-01', '2019-11-15'),
        exp_name='train_test_spilit_hyper_params',
        seed = 7777,
        encoder_type='MLPEncoder',
        num_trials=None,
        sampling=None,
        train=False,
   
        default_params =   {'N': 82,
                            'activation': 1,
                            'min_std':0.001, 
                            'dist_type':'normal',
                            'kmix':5,
                            'soft_max_type':'softmax',
                            'batch_size': 12,
                            'depth': 4,
                            'droput': 0.4,
                            'alpha': 0.5,
                            'entropy_loss': True,
                            'expansion_factor': 2,
                            'latent_size': 64,
                            'num_cosines': 64,
                            'num_head': 4,
                            'patch_size': 8,
                            'sampling': 1,
                            'kappa': 0.5583529712369528,
                            'margin': 1.0188749613200957e-06,
                            'scale': 'min_max',
                            'target_scale': 'min_max',
                            'clipping_value': 3.004622548809806,
                            'encoder_type': 'MLPEncoder',
                            'feature_start': 16,
                            'cnn_emb': False,
                            'huber_loss': True,
                            'calibration_loss': False,
                            'nrmse_loss': False,
                            'nll_loss': False,
                            'out_activation': None,
                            'droput': 0.38371718771118474,
                            'max_epochs': 800,
                            'period': '30T',
                            'rolling_window': 3,
                            'SAMPLES_PER_DAY': 48,
                            'time_varying_known_feature': ['Ghi'],
                            'time_varying_unknown_feature': ['Load-median-filter', 'Load-Ghi'],
                            'time_varying_known_categorical_feature': ['DAYOFWEEK', 'HOUR', 
                                                                       'WEEKDAY', 'WEEKEND', 
                                                                       'SATURDAY', 'SUNDAY'],
                            'categorical_dims': [52, 7, 356, 12, 32, 24, 2, 2, 2, 2],
                            'categorical_emb': False,
                            'targets': 'Load-target',
                            'window_size': 96,
                            'horizon': 48, 
                             'yearly_seasonality':"auto",
                             'weekly_seasonality':"auto",
                             'daily_seasonality':"auto",
                             'seasonality_mode':"additive",
                             'seasonality_reg':0,
                             'seasonality':True}):

        default_params.update({'encoder_type': encoder_type})

        
        #f"Hyper-{exp_name}_{encoder_type}"
        
        default_params = get_best_params(default_params, study_name =f"Hyper-{exp_name}_{encoder_type}" , seed=777)
        if sampling is not None:
            default_params.update({'sampling': sampling})
        self.encoder_type = encoder_type
        self.hparams = default_params
        self.num_trials=num_trials
        self.combine=combine,
        self.add_ghi_feature=add_ghi_feature,
        self.cleanData=classmethod,
        self.window=window,
        self.train_window=train_window,
        self.test_window=test_window,
        
        self.seed = seed,
        pl.seed_everything(seed, workers=True)
        self.hparams.update({ 'encoder_type': self.encoder_type})
        if self.hparams['seasonality']:
            self.exp_name =f'{exp_name}_with_seasonality'

        else:
            self.exp_name=exp_name
        self.train = train


    def baseline(self, trial=None, experiment=None, file_name=None):

        if trial is not None:
            hparams =  get_search_params(trial, self.hparams)
            checkpoints = Path(f"../checkpoints/{self.exp_name}/{hparams['encoder_type']}/{trial.number}")
        else:
            hparams = self.hparams
            if file_name is not None:
                checkpoints = Path(f"../checkpoints/{self.exp_name}/{hparams['encoder_type']}/{file_name}.joblib")
            else:
                checkpoints = Path(f"../checkpoints/{self.exp_name}/{hparams['encoder_type']}.joblib")
        hparams.update({'period': periods[hparams['sampling']]})
        hparams.update({'SAMPLES_PER_DAY': samples[hparams['sampling']]})
        hparams.update({ 'window_size': 2*samples[hparams['sampling']]})
        hparams.update({ 'horizon': samples[hparams['sampling']]})

        results_path = Path(f"../results/{self.exp_name}/{hparams['encoder_type']}/")
        logs = Path(f"../logs/{self.exp_name}/{hparams['encoder_type']}/")
        figures = Path(f"../figures/{self.exp_name}/{hparams['encoder_type']}/")
        figures.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        results_path.mkdir(parents=True, exist_ok=True)
        
        if experiment is None:
            experiment = DatasetObjective(hparams,self.combine, self.add_ghi_feature,  self.cleanData, self.window)



    
        
        checkpoints.mkdir(parents=True, exist_ok=True)
        
        train_loader = experiment.get_dataset(hparams, self.train_window, shufle=False, drop_last=False, test=True)
        test_loader = experiment.get_dataset(hparams,  self.test_window, shufle=False, drop_last=False, test=True)
        index = experiment.data.loc[self.test_window].iloc[96:].index
        gt = experiment.target_transformer.inverse_transform(experiment.data[[hparams['targets']]].values)
        print(f"---------------Training on {self.hparams['encoder_type']} ---------------------------")
        
        if self.hparams['encoder_type'] in ['RF', 'SVR']:
            index = experiment.data.loc[self.test_window].iloc[96:].index
           
            logs=fit_baseline(train_loader, test_loader, experiment, index, file_name, baseline=hparams['encoder_type'])

        else:
            if self.hparams['encoder_type']=='ARNET':
                logs=fit_auto_regressive_forecasting(self.train_window, self.test_window, experiment, hparams)
                

            if self.hparams['encoder_type']=='SARIMAX':
                logs=fit_sarimax_forecasting(self.train_window, self.test_window, experiment, hparams)
                
                

            if self.hparams['encoder_type']=='LinearHMM':
                logs=fit_space_state_forecasting(self.train_window, self.test_window, experiment, hparams)
                
                

        #save results
        np.save(f"../results/{self.exp_name}/{hparams['encoder_type']}/{file_name}_processed_results.npy", logs)



        #visualize results
        if self.hparams['encoder_type'] in ['RF', 'SVR', 'ARNET', 'SARIMAX']:

            fig, ax = plt.subplots(1, 1, figsize = (9,3))
            ax=plot_prediction(ax, logs['true'][:96], logs['pred'][:96], logs['index'][:96])
            fig.autofmt_xdate(rotation=90, ha='center')
            met=logs['metrics'][['nrmse', 'mae']].mean().values
            ax.set_title("NMRSE: {:.2g}, MAE: {:.2g}%".format(met[0], met[1]), fontsize=15); 
            min_y=min(gt.min(), logs['pred'].min())
            max_y=max(gt.max(), logs['pred'].max())
            ax.set_ylim(min_y, max_y)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            fig.autofmt_xdate(rotation=90, ha='center')
            fig.tight_layout()
            fig.savefig(f"../figures/{self.exp_name}/{hparams['encoder_type']}/{file_name}_confidence.pdf", dpi=480)
            plt.close()

        if self.hparams['encoder_type'] in ('LinearHMM'):
            
           
        
            fig, ax = plt.subplots(1, 1, figsize = (9,3))
            ax=plot_prediction_with_pi(ax, logs['true'][:96], logs['pred'][:96], logs['q_pred'][:96, :, 0].T, logs['index'][:96])
            met=logs['metrics'][['nrmse', 'ciwe', 'ncrps']].mean().values
            ax.set_title("NMRSE: {:.2g}, CWE: {:.3g}, CRPS: {:.3g}%".format(met[0], met[1], met[2]), fontsize=15); 
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            min_y=min(gt.min(), logs['pred'].min())
            max_y=max(gt.max(), logs['pred'].max())
            ax.set_ylim(min_y, max_y)
            fig.autofmt_xdate(rotation=90, ha='center')
            fig.tight_layout()
            fig.savefig(f"../figures/{self.exp_name}/{hparams['encoder_type']}/{file_name}_confidence.pdf", dpi=480)
            plt.close()
            


        return logs['metrics']

    def objective(self, trial=None, experiment=None, file_name=None, horizon=None):
        # Filenames for each trial must be made unique in order to access each checkpoint.

        
        if trial is not None:
            hparams =  get_search_params(trial, self.hparams)
            checkpoints = Path(f"../checkpoints/{self.exp_name}/{hparams['encoder_type']}/{trial.number}")
        else:
            hparams = self.hparams
            if file_name is not None:
                checkpoints = Path(f"../checkpoints/{self.exp_name}/{hparams['encoder_type']}/{file_name}")
            else:
                checkpoints = Path(f"../checkpoints/{self.exp_name}/{hparams['encoder_type']}")
        hparams.update({'period': periods[hparams['sampling']]})
        hparams.update({'SAMPLES_PER_DAY': samples[hparams['sampling']]})
        hparams.update({ 'window_size': 2*samples[hparams['sampling']]})
        
        if horizon is None:
            hparams.update({ 'horizon': samples[hparams['sampling']]})
        else:
            hparams.update({ 'horizon': horizon})
        

        results_path = Path(f"../results/{self.exp_name}/{hparams['encoder_type']}/")
        logs = Path(f"../logs/{self.exp_name}/{hparams['encoder_type']}/")
        figures = Path(f"../figures/{self.exp_name}/{hparams['encoder_type']}/")
        figures.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        results_path.mkdir(parents=True, exist_ok=True)
        
        if experiment is None:
            experiment = DatasetObjective(hparams,self.combine, self.add_ghi_feature,  self.cleanData, self.window)



        
    
        
        checkpoints.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoints, 
                                                                    monitor='val_mae', 
                                                                    mode="min", 
                                                                every_n_val_epochs=5,
                                                                save_top_k=2)
        if trial is not None:
            logger = DictLogger(logs,  version=trial.number)
            early_stopping = PyTorchLightningPruningCallback(trial, monitor='val_mae')
            file_name = f"{self.exp_name}_{hparams['encoder_type']}_{trial.number}"
        else:
            early_stopping = EarlyStopping(monitor="val_mae", min_delta=0.0, patience=int(hparams['max_epochs']*0.5), verbose=False, mode="min", check_on_train_epoch_end=True)
            lr_logger = LearningRateMonitor()
            logger  = TensorBoardLogger(logs,  name = f"{self.exp_name}", version = 0)
            if file_name is None:
                file_name = f"{self.exp_name}_{hparams['encoder_type']}"
            logger  = TensorBoardLogger(logs,  name = f"{file_name}", version = 0)
        
        
        trainer = pl.Trainer(logger = logger,
                        gradient_clip_val=hparams['clipping_value'],
                        max_epochs = hparams['max_epochs'],
                        callbacks=[checkpoint_callback,early_stopping, lr_logger],
                        gpus=1,
                        deterministic=True,
                        stochastic_weight_avg=False,
                        weights_summary=None,
                        #precision=16,
                        resume_from_checkpoint=get_latest_checkpoint(checkpoints)
                        )
        
        #train_loader = experiment.get_dataset(hparams, slice(self.train_window[0][0], self.train_window[0][1]), shufle=True)
       # test_loader = experiment.get_dataset(hparams, slice(self.test_window[0][0], self.test_window[0][1]), shufle=False)


        train_loader = experiment.get_dataset(hparams, self.train_window, shufle=True, test=False, drop_last=True)
        val_loader = experiment.get_dataset(hparams,  self.test_window, shufle=False, test=False, drop_last=True)
        test_loader = experiment.get_dataset(hparams,  self.test_window, shufle=False, test=True, drop_last=False)
        
            
        
        
            
        n_channels = next(iter(train_loader))[0].size(-1)
        if hparams['encoder_type'] in  ['GRUBaseline', 'LSTMBaseline']:
            cell_type = 'GRU' if hparams['encoder_type']=='GRUBaseline' else 'LSTM'
            network = RNNBaseline(input_size=n_channels, 
                            hidden_size=hparams['latent_size'], 
                            num_layers=hparams['depth'], 
                            bidirectional=False, 
                            dropout=hparams['droput'], 
                            activation=activations[hparams['activation']], 
                            out_size=hparams['horizon'], 
                            context_size=hparams['window_size']+hparams['horizon'], 
                            cell_type = cell_type)
            model = RNNBaselineModel_pil(network,  hparams, lr=1e-3)



        elif hparams['encoder_type'] in ['GRUGauss', 'LSTGauss']:
            cell_type = 'GRU' if hparams['encoder_type']=='GRUGauss' else 'LSTM'
          
            network = RNNGausModel(input_size=n_channels, 
                            hidden_size=hparams['latent_size'], 
                            num_layers=hparams['depth'], 
                            bidirectional=False, 
                            dropout=hparams['droput'], 
                            activation=activations[hparams['activation']], 
                            out_size=hparams['horizon'], 
                            context_size=hparams['window_size']+hparams['horizon'])
            model = RNNGaussModel_pil(network,  hparams, 1e-3)

        elif hparams['encoder_type'] in ['GRUMDN', 'LSTMMDN']:
            cell_type = 'GRU' if hparams['encoder_type']=='GRUMDN' else 'LSTM'
            network = RNNMDNModel(input_size=n_channels, 
                            hidden_size=hparams['latent_size'],
                            latent_size=hparams['latent_size']*2, 
                            num_layers=hparams['depth'], 
                            bidirectional=False, 
                            dropout=hparams['droput'], 
                            activation=activations[hparams['activation']], 
                            out_size=hparams['horizon'], 
                            context_size=hparams['window_size']+hparams['horizon'], 
                            cell_type = cell_type, 
                            kmix=hparams['kmix'],   
                            min_std = hparams['min_std'], 
                            dist_type=hparams['dist_type'],
                            soft_max_type=hparams['soft_max_type'],
                            alpha=hparams['alpha'])
            model = RNNMDNModel_pil(network,  hparams, lr=1e-3)

        else:
        
            network = FPQSeq2Q(n_channels = n_channels,
                                                latent_size = hparams['latent_size'], 
                                                out_size = hparams['horizon'], 
                                                context_size = hparams['window_size'], 
                                                dropout = hparams['droput'],
                                                N = hparams['N'],
                                                huber_loss = hparams['huber_loss'],
                                                calibration_loss = hparams['calibration_loss'],
                                                entropy_loss = hparams['entropy_loss'],
                                                nrmse_loss = hparams['nrmse_loss'],
                                                out_activation=hparams["out_activation"],
                                                nll_loss = hparams['nll_loss'],
                                                patch_size= hparams['patch_size'], 
                                                expansion_factor =  hparams['expansion_factor'],        
                                                depth = hparams['depth'],
                                                num_head=hparams['num_head'],
                                                encoder_type=hparams['encoder_type'],
                                                num_cosines=hparams['num_cosines'],
                                                activation=activations[hparams['activation']])
                
                                            
                    
            model   = model_pil(network, hparams, lr=1e-3)

        print(f"---------------Training on {self.hparams['encoder_type']} ---------------------------")
        if self.train:
            start_time = default_timer()
            trainer.fit(model, train_loader, val_loader)
            train_walltime = default_timer() - start_time
        else:
            train_walltime = 0.0
            
        
    
        index = experiment.data.loc[self.test_window].iloc[96:].index
        path_best_model = get_latest_checkpoint(checkpoints)
        checkpoint      = torch.load(path_best_model)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        features, target = test_loader.dataset.tensors
        
        if hparams['encoder_type'] in  ['GRUBaseline', 'LSTMBaseline', 'GRUGauss', 'LSTGauss', 'GRUMDN', 'LSTMMDN']:
            true = experiment.target_transformer.inverse_transform(target.numpy())
            start_time = default_timer()
            pred, q_pred, n_samples= model(features)
            test_walltime = default_timer() - start_time
            pred = experiment.target_transformer.inverse_transform(pred.numpy())
            
            q_pred = q_pred.numpy()
            N, M, T = q_pred.shape
            q_pred = q_pred.reshape(N*M, T)
            q_pred = experiment.target_transformer.inverse_transform(q_pred)
            q_pred = q_pred.reshape(N, M, T)
            
            
            M, N, T = n_samples.shape
            n_samples = n_samples.reshape(N*M, T)
            n_samples = experiment.target_transformer.inverse_transform(n_samples.numpy())
            n_samples = n_samples.reshape(M, N, T)
            tau_hats=np.array(q)
            tau_hats=torch.from_numpy(tau_hats[None, :, None]).expand_as(torch.from_numpy(q_pred)).numpy()
            
        else:
            start_time = default_timer()
            true, pred, q_pred, tau_hats = get_predictions(model.model, features, target, experiment)
            test_walltime = default_timer() - start_time
            n_samples=None
            #results = model.predict(model, test_loader)
       
        

        
            

        pred = pred.reshape(-1, 1)
        true = true.reshape(-1, 1)
        #pred = results["pred"].data.numpy().reshape(-1, 1)
        #q_pred = results["q_pred"].data.numpy()
        #true =results["true"].data.numpy().reshape(-1, 1)
        #tau_hats = results['tau_hat'].data.numpy()

        #score=cwi_score(torch.tensor(true), torch.tensor(q_pred))

            
        '''
        if experiment.hparams['target_scale']=='log_scaler':
            pred = np.expm1(pred)
            true = np.expm1(true)
            q_pred = np.expm1(q_pred)
            gt = np.expm1(experiment.data[hparams['targets']].values)

        else:
            pred = experiment.target_transformer.inverse_transform(pred)
            true = experiment.target_transformer.inverse_transform(true)
            N, M, T = q_pred.shape
            q_pred = q_pred.reshape(N*M, T)
            q_pred = experiment.target_transformer.inverse_transform(q_pred)
            q_pred = q_pred.reshape(N, M, T)
            target_range=true.max()-true.min()
    

        if tau_hats.ndim==2:
            std_pred =(tau_hats[:,:, None]*q_pred).std(axis=1)
        else:
            std_pred =(tau_hats*q_pred).std(axis=1)
        '''

        q_pred=np.hstack([h for h in q_pred]).T[:,:,None]
        tau_hats=np.hstack([h for h in tau_hats]).T[:,:,None]
        
        target_range = true.max()-true.min()
        gt = experiment.target_transformer.inverse_transform(experiment.data[[hparams['targets']]].values)
        
        #target_range=gt.max() - gt.min()
        metrics=[]
        #samples[:, i:i+48,0]
        for i in range(0, len(pred), 48):
           
            metric =get_daily_metrics(pred[i:i+48, 0].T, true[i:i+48, 0].T, q_pred[i:i+48,:, 0].T,
                                     target_range, 2*true[i:i+48, 0].T.std()/target_range, samples=None, tau = tau_hats[i:i+48,:, 0].T,
                                    alpha=1.0)
            
            metrics.append(metric)
            #print(index[i:i+48][0])
        metrics = pd.concat(metrics)
        #crps = eval_crps(torch.tensor(q_pred).permute(1,0,2), torch.tensor(true))
        #metric=get_overall_metrics(true, pred, q_pred, tau_hats, crps, target_range)
           
        metrics['train_time']=train_walltime
        metrics['test_time']=test_walltime
      
        logs  = {"pred": pred,  "tau_hat":tau_hats, "q_pred":q_pred,  "true":true, 'target_range':target_range, 
             'metrics':metrics, 'index':index,  "feats":features.data.numpy(), "samples":n_samples}

        print(pd.DataFrame(logs['metrics'].mean()).T[[  'mae', 'nrmse',  'ncrps',  'pic',  'nmpi', 'ciwe',  'ciwf', 'corr']].round(2))
           
        np.save(f"../results/{self.exp_name}/{hparams['encoder_type']}/{file_name}_processed_results.npy", logs)
        fig, ax = plt.subplots(1, 1, figsize = (9,3))
        ax=plot_prediction_with_pi(ax, logs['true'][:96], logs['pred'][:96], logs['q_pred'][:96, :, 0].T, logs['index'][:96])
        met=logs['metrics'][['nrmse', 'ciwe', 'ncrps']].mean().values
        ax.set_title("NMRSE: {:.2g}, CWE: {:.3g}, CRPS: {:.3g}%".format(met[0], met[1], met[2]), fontsize=15); 
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        min_y=min( gt.min(), logs['pred'].min())
        max_y=max(gt.max(), logs['pred'].max())
        ax.set_ylim(min_y, max_y)
        fig.autofmt_xdate(rotation=90, ha='center')
        fig.tight_layout()
        fig.savefig(f"../figures/{self.exp_name}/{hparams['encoder_type']}/{file_name}_confidence.pdf", dpi=480)
        plt.close()



        if trial is not None:
            return logger.metrics[-1]['val_mae_epoch']
        else:
            return logs['metrics']

    def bactesting(self, baseline=False, experiment=None):
        """[Conduct backtesting cross-validation]

        Args:
            baseline (bool, optional): weather to run a baseline or FPSeQ model. Defaults to False.

        Returns:
            [type]: [description]
        """
        
        test_horizon=int(self.hparams['SAMPLES_PER_DAY']*31*2) 
        train_window_size=int(self.hparams['SAMPLES_PER_DAY']*31*6)
        step_size=int(self.hparams['SAMPLES_PER_DAY']*31)
        if experiment is None:
            experiment = DatasetObjective(self.hparams,self.combine, self.add_ghi_feature,  self.cleanData, self.window)
        

       
        duration= experiment.data.shape[0]
        max_window = duration - test_horizon
        generator = bactesting_expanding(duration, test_horizon, train_window_size, step_size, max_window, period=self.hparams['SAMPLES_PER_DAY'])
        exp_name = self.exp_name
        cross_validion = 1
        metrics_list=[]
        metrics_spilit={}

        for train_index, test_index in generator:
            #get slice of start and end datetime for each set
            
            if cross_validion >0:
                window = slice(experiment.data.index[train_index][0],  experiment.data.index[test_index][-1])
                train_window = slice(experiment.data.index[train_index][0], experiment.data.index[train_index][-1])
                test_window  = slice(experiment.data.index[test_index][0], experiment.data.index[test_index][-1])
                
                exp_name_full = f"cross_validation_{cross_validion}"
                self.train_window=train_window
                self.test_window=test_window
                self.window=window
                print(f"---------------Backtesting expanding-{cross_validion} Cross validation Training --------------------------")
                print("")
                print(f"Train_window: {train_window}")
                print("")
                print(f"Test_window: {test_window}")
                print("")
                
                if baseline:
                    #if (cross_validion==2) and (self.hparams['encoder_type'] =='LinearHMM'):
                    #    continue
                    #else:
                    metric = self.baseline(experiment=experiment, file_name=exp_name_full)
                else:
                    metric = self.objective(experiment=experiment, file_name=exp_name_full)

                #print(pd.DataFrame(metric.mean()).T)
                print("")
                metrics_list.append(pd.DataFrame(metric.mean()).T)
                metrics_spilit[cross_validion] = metric
            cross_validion+=1

        

        pd_metrics = pd.concat(metrics_list)
        x_labels = ["Split-"+str(i+1) for i in range(cross_validion-1)]
        fig, ax = plt.subplots(1,1, figsize=(9,3))
        ax=plot_boxplot_per_encoder(ax, metrics_spilit, 'nrmse', 'NRMSE')
        ax.set_title(f"{self.encoder_type} {pd_metrics['nrmse'].mean():.{3}f} $\pm$ {pd_metrics['nrmse'].std():.{3}f}")
        plt.xticks(range(1, len(x_labels)+1), x_labels, rotation=0, fontsize=12);
        fig.savefig(f"../figures/{self.exp_name}/NRMSE_{self.encoder_type}.pdf", dpi=480)
        plt.close()

       
        fig, ax = plt.subplots(1,1, figsize=(9,3))
        ax=plot_boxplot_per_encoder(ax, metrics_spilit, 'corr', 'CORR')
        ax.set_title(f"{self.encoder_type} {pd_metrics['corr'].mean():.{3}f} $\pm$ {pd_metrics['corr'].std():.{3}f}")
        plt.xticks(range(1, len(x_labels)+1), x_labels, rotation=0, fontsize=12);
        fig.savefig(f"../figures/{self.exp_name}/CORR_{self.encoder_type}.pdf", dpi=480)
        plt.close()

        


        fig, ax = plt.subplots(1,1, figsize=(9,3))
        ax=plot_boxplot_per_encoder(ax, metrics_spilit, 'mae', 'MAE')
        ax.set_title(f"{self.encoder_type} {pd_metrics['mae'].mean():.{3}f} $\pm$ {pd_metrics['mae'].std():.{3}f}")
        plt.xticks(range(1, len(x_labels)+1), x_labels, rotation=0, fontsize=12);
        plt.ylim(0, 1)
        fig.savefig(f"../figures/{self.exp_name}/MAE_{self.encoder_type}.pdf", dpi=480)
        plt.close()

        if self.hparams['encoder_type']  in ['GRUEncoder', 'MLPEncoder', 'UNETEncoder', 'LinearHMM']:
            fig, ax = plt.subplots(1,1, figsize=(9,3))
            ax=plot_boxplot_per_encoder(ax, metrics_spilit, 'pic', 'PICP')
            ax.set_title(f"{self.encoder_type} {pd_metrics['pic'].mean():.{3}f} $\pm$ {pd_metrics['pic'].std():.{3}f}")
            plt.xticks(range(1, len(x_labels)+1), x_labels, rotation=0, fontsize=12);
            fig.savefig(f"../figures/{self.exp_name}/PICP_{self.encoder_type}.pdf", dpi=480)
            plt.close()
            
            
            
            fig, ax = plt.subplots(1,1, figsize=(9,3))
            ax=plot_boxplot_per_encoder(ax, metrics_spilit, 'nmpi', 'NMPI')
            ax.set_title(f"{self.encoder_type} {pd_metrics['nmpi'].mean():.{3}f} $\pm$ {pd_metrics['nmpi'].std():.{3}f}")
            plt.xticks(range(1, len(x_labels)+1), x_labels, rotation=0, fontsize=12);
            fig.savefig(f"../figures/{self.exp_name}/NMPI_{self.encoder_type}.pdf", dpi=480)
            plt.close()


            fig, ax = plt.subplots(1,1, figsize=(9,3))
            ax=plot_boxplot_per_encoder(ax, metrics_spilit, 'ciwe', 'CWIE')
            ax.set_title(f"{self.encoder_type} {pd_metrics['ciwe'].mean():.{3}f} $\pm$ {pd_metrics['ciwe'].std():.{3}f}")
            plt.ylim(0, 1)
            plt.xticks(range(1, len(x_labels)+1), x_labels, rotation=0, fontsize=12);
            fig.savefig(f"../figures/{self.exp_name}/CIWNMRSE_{self.encoder_type}.pdf", dpi=480)
            plt.close()

            fig, ax = plt.subplots(1,1, figsize=(9,3))
            ax=plot_boxplot_per_encoder(ax, metrics_spilit, 'ciwf', 'CWIQF')
            ax.set_title(f"{self.encoder_type} {pd_metrics['ciwf'].mean():.{3}f} $\pm$ {pd_metrics['ciwf'].std():.{3}f}")
            plt.xticks(range(1, len(x_labels)+1), x_labels, rotation=0, fontsize=12);
            plt.ylim(0, 1)
            fig.savefig(f"../figures/{self.exp_name}/CWIQF_{self.encoder_type}.pdf", dpi=480)
            plt.close()

           

            fig, ax = plt.subplots(1,1, figsize=(9,3))
            ax=plot_boxplot_per_encoder(ax, metrics_spilit, 'ncrps', 'NCRSP')
            ax.set_title(f"{self.encoder_type} {pd_metrics['ncrps'].mean():.{3}f} $\pm$ {pd_metrics['ncrps'].std():.{3}f}")
            plt.xticks(range(1, len(x_labels)+1), x_labels, rotation=0, fontsize=12);
            fig.savefig(f"../figures/{self.exp_name}/NCRSP_{self.encoder_type}.pdf", dpi=480)
            plt.close()


            fig, ax = plt.subplots(1,1, figsize=(9,3))
            ax=plot_boxplot_per_encoder(ax, metrics_spilit, 'smape', 'SMAPE')
            ax.set_title(f"{self.encoder_type} {pd_metrics['smape'].mean():.{3}f} $\pm$ {pd_metrics['smape'].std():.{3}f}")
            plt.xticks(range(1, len(x_labels)+1), x_labels, rotation=0, fontsize=12);
            fig.savefig(f"../figures/{self.exp_name}/SMAPE_{self.encoder_type}.pdf", dpi=480)
            plt.close()
            




            


        return pd_metrics, metrics_spilit
        

        
    


    def tune_params(self):
        """Run hyper-parameter tuning
        """
        Path(f"Hyper-{self.exp_name}_{self.encoder_type}.db").touch(exist_ok=True)
        study = run_study(num_trials=self.num_trials, objective=self.objective, seed=self.seed, study_name = f"Hyper-{self.exp_name}_{self.encoder_type}")

    def train_spilit(self, file_name:str, horizon:int):
        """[summary]

        Args:
            file_name (str): file name to specify running experiment
            horizon (int): forecasted points depending on the data resolution

        Returns:
            [pd.dataframe]: dataframe with metrics
        """
        print(f"---------------Train test spilit-{file_name}  Training --------------------------")
        metric=self.objective(file_name=file_name, horizon=horizon)
        return metric




def run_hyperparams(window=slice('2019-03', '2020-09'), 
                    train_window=slice('2019-03', '2020-09'),
                    test_window=slice('2019-10', '2019-11-15'),
                    seed = 7777, num_trials=300, 
                    train=True, 
                    exp_name='train_test_spilit_hyperparams_20210711_FPQMixerForecast', 
                    encoders = [ "MLPEncoder"]):
    """[summary]

    Args:
        window ([type], optional): The data section used for this experiment. Defaults to slice('2019-03', '2020-09').
        train_window ([type], optional): data sample used for training. Defaults to slice('2019-03', '2020-09').
        test_window ([type], optional): data sample used for testing. Defaults to slice('2019-10', '2019-11-15').
        seed (int, optional): seed for experiment reproducibility. Defaults to 7777.
        num_trials (int, optional): Number of trilas. Defaults to 300.
        train (bool, optional): wether to train the model should be True for this function. Defaults to True.
        exp_name (str, optional): [description]. Defaults to 'train_test_spilit_hyperparams_20210711_FPQMixerForecast'.
        encoders (list, optional): [description]. Defaults to [ "MLPEncoder"].
    """
    
    
    results_path = Path(f"../results/{exp_name}/")
    figures = Path(f"../figures/{exp_name}/")
    figures.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)


    for encoder in encoders:
        
        exp=Experiment(window=window, train_window=train_window,
                        test_window=test_window,
                        exp_name=exp_name, 
                        seed = seed, 
                        encoder_type=encoder, 
                        num_trials=num_trials, 
                        train=True)
        exp.hparams.update({ 'max_epochs': 100})
        exp.tune_params()




def run_train_spilit(window=slice('2019-01', '2020-09'),
                exp_name='train_test_net_load_forecasting', 
                    encoders =  [ "MLPEncoder", "GRUEncoder", "UNETEncoder"], 
                    train='True',
                    seed=777,
                    max_epochs=200):
                    
       
 
    results_path = Path(f"../results/{exp_name}/")
    figures = Path(f"../figures/{exp_name}/")
    figures.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    
    metrics_all = {}
    pd_metrics_all = {}
    sampling = 1
    

    train_window=slice('2019-01', '2020-08')
    test_window=slice('2020-09', '2020-10')

    for encoder in encoders:
        for horizon in [48]:
            exp=Experiment(window=window,  
           train_window=train_window, test_window=test_window, exp_name=exp_name, seed = seed, encoder_type=encoder, sampling=sampling, train=train)
            if max_epochs is not None:
                exp.hparams.update({ 'max_epochs': max_epochs})

            if exp.hparams['seasonality']:
                file_name=f'{horizons[48]}_with_seasonality_forecasting'
            else:
                file_name=f'{horizons[48]}_forecasting'
    

            if encoder in ['LinearHMM', 'SARIMAX', 'ARNET',  'RF', 'SVR'  ]:
                metric = exp.baseline(file_name=file_name)
            else:
                metric = exp.train_spilit(file_name=file_name, horizon=horizon)
            
            #print(pd.DataFrame(metric.mean()).T)
            metrics_all[sampling]= metric
        pd_metrics_all[encoder]= metrics_all

    
    np.save(f"../results/{exp_name}/allmetrics.npy", pd_metrics_all)
        
            

    
def run_backtesting(window=slice('2019-03 01 00:00:00', '2020-09-28 23:00:00'), 
                    seed=777,
                    max_epochs=200,
                    train=True,
                    exp_name='backtesting_expanding_cross_net_load_forecasting-v2', 
                    encoders = ['LinearHMM',  'GRUGauss', "MLPEncoder", "GRUEncoder", "UNETEncoder",  'RF', 'SVR',  'ARNET']):

                    # , 'LinearHMM',  'GRUMDN',  'GRUBaseline', 'LinearHMM', 'LinearHMM' 'RF', 'SVR' "MLPEncoder", "GRUEncoder", "UNETEncoder", 'SARIMAX', 'ARNET',  'RF', 'SVR'
    """[summary]

    Args:
        window ([type], optional): [description]. Defaults to slice('2019-03', '2020-09').
        seed (int, optional): [description]. Defaults to 777.
        max_epochs (int, optional): [description]. Defaults to 200.
        train (bool, optional): [description]. Defaults to True.
        exp_name (str, optional): [description]. Defaults to 'backtesting_expanding_cross_validation_20210711'.
        encoders (list, optional): [description]. Defaults to ["GRUEncoder", "MLPEncoder", "MLPMixerEncoder", "UNETEncoder", 'RF', 'SVR'].
    """

    
    
    pd_metrics_all = {}
    pd_metrics_spilit = {}
    results_path = Path(f"../results/{exp_name}/")
    figures = Path(f"../figures/{exp_name}/")
    figures.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    combine=True
    add_ghi_feature=True
    cleanData=True
    
    for encoder in encoders:
        #train=False if encoder in ['LSTMBaseline', 'GRUEncoder', "MLPEncoder", "MLPMixerEncoder", "UNETEncoder"] else True
        exp=Experiment(window=window, exp_name=exp_name, seed = seed, encoder_type=encoder, train=train)
        experiment = DatasetObjective(exp.hparams,combine, add_ghi_feature,  cleanData, exp.window)
        #experiment =None
       

        if max_epochs is not None:
            exp.hparams.update({ 'max_epochs': max_epochs})
        if encoder in ('RF', 'SVR', 'LinearHMM', 'ARNET', 'SARIMAX'):
            baseline=True
        else:
            baseline=False
        pd_metrics, metrics_spilit = exp.bactesting(baseline=baseline, experiment=experiment)
        pd_metrics_all[encoder]= pd_metrics
        pd_metrics_spilit[encoder]=metrics_spilit
        

        if encoder not in ['RF', 'SVR', 'ARNET','SARIMAX' ]:

            print(encoder)
            print("MAE,  NRMSE,    NCRSP,   PIC,  NMPI, CWE,  CWF  SMAPE  CORR")
            print(f"{pd_metrics['mae'].mean():.{2}f} $\pm$ {pd_metrics['mae'].std():.{2}f}, \
            {pd_metrics['nrmse'].mean():.{2}f} $\pm$ {pd_metrics['nrmse'].std():.{2}f}, \
            {pd_metrics['ncrps'].mean():.{2}f} $\pm$ {pd_metrics['ncrps'].std():.{2}f}, \
            {pd_metrics['pic'].mean():.{2}f} $\pm$ {pd_metrics['pic'].std():.{2}f}, \
            {pd_metrics['nmpi'].mean():.{2}f} $\pm$ {pd_metrics['nmpi'].std():.{2}f}, \
            {pd_metrics['ciwe'].mean():.{2}f} $\pm$ {pd_metrics['ciwe'].std():.{2}f}, \
            {pd_metrics['ciwf'].mean():.{2}f} $\pm$ {pd_metrics['ciwf'].std():.{2}f} , \
            {pd_metrics['corr'].mean():.{3}f} $\pm$ {pd_metrics['corr'].std():.{3}f}")
            

        else:
            print(encoder)
            print("MAAPE, MAE,  NRMSE,    CORR")
            print(f"{pd_metrics['mae'].mean():.{2}f} $\pm$ {pd_metrics['mae'].std():.{2}f}, \
        {pd_metrics['nrmse'].mean():.{2}f} $\pm$ {pd_metrics['nrmse'].std():.{2}f}, \
        {pd_metrics['corr'].mean():.{2}f} $\pm$ {pd_metrics['corr'].std():.{2}f}")

    
    np.save(f"../results/{exp_name}/allmetrics.npy", pd_metrics_all)
    np.save(f"../results/{exp_name}/split_allmetrics.npy", pd_metrics_spilit)


    fig, ax = plt.subplots(1,1, figsize=(9,3))
    ax=plot_boxplot_per_encoder(ax, pd_metrics_all, 'train_time', 'TRA-TIME')
    fig.savefig(f"../figures/{exp_name}/TRA-TIME.pdf", dpi=480)
            
            
    fig, ax = plt.subplots(1,1, figsize=(9,3))
    ax=plot_boxplot_per_encoder(ax, pd_metrics_all, 'test_time', 'TEST-TIME')
    fig.savefig(f"../figures/{exp_name}/TEST-TIME.pdf", dpi=480)
            
    fig, ax = plt.subplots(1,1, figsize=(9,3))
    ax=plot_boxplot_per_encoder(ax, pd_metrics_all, 'nrmse', 'NRMSE')
    fig.savefig(f"../figures/{exp_name}/NRMSE.pdf", dpi=480)

        
    fig, ax = plt.subplots(1,1, figsize=(9,3))
    ax=plot_boxplot_per_encoder(ax, pd_metrics_all, 'smape', 'SMAPE')
    fig.savefig(f"../figures/{exp_name}/SMAAPE.pdf", dpi=480)
        
    
    if encoder not in ['RF', 'SVR', 'ARNET','SARIMAX' ]:
        fig, ax = plt.subplots(1,1, figsize=(9,3))
        ax=plot_boxplot_per_encoder(ax, pd_metrics_all, 'pic', 'PICP')
        fig.savefig(f"../figures/{exp_name}/PICP.pdf", dpi=480)
        

        fig, ax = plt.subplots(1,1, figsize=(9,3))
        ax=plot_boxplot_per_encoder(ax, pd_metrics_all, 'nmpi', 'NMPI')
        fig.savefig(f"../figures/{exp_name}/NMPI.pdf", dpi=480)

        fig, ax = plt.subplots(1,1, figsize=(9,3))
        ax=plot_boxplot_per_encoder(ax, pd_metrics_all, 'ciwe', 'CIWNRMSE')
        fig.savefig(f"../figures/{exp_name}/CIWNRMSE.pdf", dpi=480)


        #fig, ax = plt.subplots(1,1, figsize=(9,3))
        #ax=plot_boxplot_per_encoder(ax, pd_metrics_all, 'train_time', 'TRA-TIME')
        #fig.savefig(f"../figures/{exp_name}/TRA-TIME.pdf", dpi=480)
            
            
        #fig, ax = plt.subplots(1,1, figsize=(9,3))
        #ax=plot_boxplot_per_encoder(ax, pd_metrics_all, 'test_time', 'TEST-TIME')
        #fig.savefig(f"../figures/{exp_name}/TEST-TIME.pdf", dpi=480)
            
            

       
if __name__ == "__main__":
    #run_train_spilit()
    
    run_backtesting()
    #run_train_spilit()
    #run_hyperparams()
    #run_real_time()

    
       
    
    
  
  
    