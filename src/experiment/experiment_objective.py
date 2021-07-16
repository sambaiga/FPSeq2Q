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
from utils.visual_functions import plot_intervals_ordered, plot_xy, plot_calibration, plot_boxplot_per_encoder
from utils.data import DatasetObjective
from experiment.backtesting import bactesting_sliding, bactesting_expanding
from experiment.baseline_experiment import  fit_baseline, get_metrics
from utils.optimisation import get_search_params, get_best_params, run_study
from net.model import FPQSeq2Q
from net.loss_functions import cwi_score
from net.model_pil import model_pil
from net.layers import Swish
from net.utils import DictLogger
from utils.data import TimeSeriesDataset
from  net.utils import get_latest_checkpoint
from net.metrics import get_metrics_dataframe
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
        window=slice('2019-03-01', '2019-11-15'),
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
                            'batch_size': 32,
                            'depth': 4,
                            'droput': 0.4,
                            'entropy_loss': True,
                            'expansion_factor': 2,
                            'latent_size': 64,
                            'num_cosines': 512,
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
                            'max_epochs': 200,
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
        
        train_loader = experiment.get_dataset(hparams, self.train_window, shufle=True)
        test_loader = experiment.get_dataset(hparams,  self.test_window, shufle=False)
        
        
        print(f"---------------Training on {self.hparams['encoder_type']} ---------------------------")
        pred, true=fit_baseline(train_loader, test_loader, file_name, baseline=hparams['encoder_type'])

        if experiment.hparams['target_scale']=='log_scaler':
            pred = np.expm1(pred)
            true = np.expm1(true)
            gt = np.expm1(experiment.data[hparams['targets']].values)

        else:

           
            pred = experiment.target_transformer.inverse_transform(pred)
          
            true = experiment.target_transformer.inverse_transform(true)
         
            gt = experiment.target_transformer.inverse_transform(experiment.data[[hparams['targets']]].values)


        target_range=gt.max() - gt.min()
        
        metrics = get_metrics_dataframe(true, pred, q_pred=None, tau_hat=None, a_step=1, R=target_range, true_nmpic=None)

        print(pd.DataFrame(metrics.mean()).T[['nrmse'   , 'corr'   ,    'maap']])
        
        
 
        results=dict(true=true, pred=pred, metrics=metrics)
        np.save(f"../results/{self.exp_name}/{hparams['encoder_type']}/{file_name}.npy", results)
        horizon = hparams['horizon']
        
        index = np.arange(1, 13)
        fig, axs = plt.subplots(3, 4, figsize=(10,6), sharey=True, sharex=True)
        axs = axs.ravel()
        for i, ax in enumerate(axs):
            day = int(horizon*index[i])
            x  = np.arange(day)
            h1 = ax.plot(true[day], ".", mec="#ff7f0e", mfc="None")
            h2 = ax.plot(pred[day],   '.-',  c="#1f77b4", alpha=0.8)
    
            
        ax.set_ylabel('Power $(KW)$')
        ax.autoscale(tight=True)
        ax.set_ylim(true.min(), true.max())
        lines =[h1[0], h2[0]]
        labels = ["True", "Pred mean"]
        ax.legend(lines, labels, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        fig.tight_layout()
        fig.savefig(f"../figures/{self.exp_name}/{hparams['encoder_type']}/{file_name}_confidence.pdf", dpi=480)
        plt.close()

        return metrics

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
                        callbacks=[checkpoint_callback,early_stopping],
                        gpus=1,
                        deterministic=True,
                        stochastic_weight_avg=False,
                        weights_summary=None,
                        #precision=16,
                        resume_from_checkpoint=get_latest_checkpoint(checkpoints)
                        )
        
        #train_loader = experiment.get_dataset(hparams, slice(self.train_window[0][0], self.train_window[0][1]), shufle=True)
       # test_loader = experiment.get_dataset(hparams, slice(self.test_window[0][0], self.test_window[0][1]), shufle=False)


        train_loader = experiment.get_dataset(hparams, self.train_window, shufle=True)
        test_loader = experiment.get_dataset(hparams,  self.test_window, shufle=False)
        
            
        
        
            
        n_channels = next(iter(train_loader))[0].size(-1)

        
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
                                            activation=activations[hparams['activation']])
            
                                        
                
        model   = model_pil(network, hparams)
        print(f"---------------Training on {self.hparams['encoder_type']} ---------------------------")
        if self.train:
            start_time = default_timer()
            trainer.fit(model, train_loader, test_loader)
            train_walltime = default_timer() - start_time
        else:
            train_walltime = 0.0
            
        
        
        
        path_best_model = get_latest_checkpoint(checkpoints)
        checkpoint      = torch.load(path_best_model)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        start_time = default_timer()
        results = model.predict(model, test_loader)
        test_walltime = default_timer() - start_time
        

        if self.hparams['encoder_type']=='LSTMBaseline':
            pred = results["pred"].data.numpy()
            true =results["true"].data.numpy()

            if experiment.hparams['target_scale']=='log_scaler':
                pred = np.expm1(pred)
                true = np.expm1(true)
                gt = np.expm1(experiment.data[hparams['targets']].values)

            else:
                pred = experiment.target_transformer.inverse_transform(pred)
                true = experiment.target_transformer.inverse_transform(true)
                gt = experiment.target_transformer.inverse_transform(experiment.data[[hparams['targets']]].values)


            target_range=gt.max() - gt.min()
            metric = get_metrics_dataframe(true, pred, q_pred=None, tau_hat=None, a_step=1, R=target_range, true_nmpic=None)
            metric['train_time']=train_walltime
            metric['test_time']=test_walltime
            results['metrics']=metric
            np.save(f"../results/{self.exp_name}/{hparams['encoder_type']}/{file_name}.npy", results)
            print(pd.DataFrame(metric.mean()).T[[ 'maap', 'nrmse'   , 'corr'  ]].round())
            

            horizon = hparams['horizon']
        
            index = np.arange(1, 13)
            fig, axs = plt.subplots(3, 4, figsize=(10,6), sharey=True, sharex=True)
            axs = axs.ravel()
            for i, ax in enumerate(axs):
                day = int(horizon*index[i])
                x  = np.arange(day)
                h1 = ax.plot(true[day], ".", mec="#ff7f0e", mfc="None")
                h2 = ax.plot(pred[day],   '.-',  c="#1f77b4", alpha=0.8)
        
                
            ax.set_ylabel('Power $(KW)$')
            ax.autoscale(tight=True)
            ax.set_ylim(true.min(), true.max())
            lines =[h1[0], h2[0]]
            labels = ["True", "Pred mean"]
            ax.legend(lines, labels, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
            fig.tight_layout()
            fig.savefig(f"../figures/{self.exp_name}/{hparams['encoder_type']}/{file_name}_confidence.pdf", dpi=480)
            plt.close()
            

        else:
            pred = results["pred"].data.numpy()
            q_pred = results["q_pred"].data.numpy()
            true =results["true"].data.numpy()
            tau_hat = results['tau_hat'].data.numpy()

            score=cwi_score(torch.tensor(true), torch.tensor(q_pred))

            

            if experiment.hparams['target_scale']=='log_scaler':
                pred = np.expm1(pred)
                true = np.expm1(true)
                q_pred = np.expm1(q_pred)
                gt = np.expm1(experiment.data[hparams['targets']].values)

            else:
                pred = experiment.target_transformer.inverse_transform(pred)
            
                true = experiment.target_transformer.inverse_transform(true)
    
                
                gt = experiment.target_transformer.inverse_transform(experiment.data[[hparams['targets']]].values)
                for k in range(q_pred.shape[1]):
                    for j in range(q_pred.shape[2]):
                        q_pred[:,k,j] = experiment.target_transformer.inverse_transform(q_pred[:,k,j][:, None]).flatten()

            if tau_hat.ndim==2:
                std_pred =(tau_hat[:,:, None]*q_pred).std(axis=1)
            else:
                 std_pred =(tau_hat*q_pred).std(axis=1)
            target_range=gt.max() - gt.min()
                    

            #metric=get_metrics_dataframe(true, pred,  q_pred, tau_hat,  experiment.hparams['SAMPLES_PER_DAY'], target_range)
            metric=get_metrics_dataframe(true, pred,  q_pred, tau_hat,  1, target_range)
            
            
            metric['train_time']=train_walltime
            metric['test_time']=test_walltime
            metric['score']=score.data.cpu().numpy()
           
            print(pd.DataFrame(metric.mean()).T[[ 'maap', 'nrmse',   'pic',  'nmpi', 'ciwrmse',  'corr']].round(3))
            print('')


            horizon = hparams['horizon']
            
            index = np.arange(1, 13)
            fig, axs = plt.subplots(3, 4, figsize=(10,6), sharey=True, sharex=True)
            axs = axs.ravel()
            for i, ax in enumerate(axs):
                day = int(horizon*index[i])
                ax, lines, labels=plot_xy(ax, pred[day], true[day], q_pred[day,:,:], true.min(), true.max())
            ax.legend(lines, labels, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
            fig.tight_layout()
            fig.savefig(f"../figures/{self.exp_name}/{hparams['encoder_type']}/{file_name}_confidence.pdf", dpi=480)
            plt.close()


                
            fig, axs = plt.subplots(3, 4, figsize=(12,8), sharey=True, sharex=True)
            axs = axs.ravel()
            for i, ax in enumerate(axs):
                day = int(horizon*index[i])
                ax=plot_intervals_ordered(ax, pred[day], std_pred[day], true[day])
            fig.tight_layout()
            fig.savefig(f"../figures/{self.exp_name}/{hparams['encoder_type']}/{file_name}_intervals.pdf", dpi=480)
            plt.close()

                
                
            fig, axs = plt.subplots(3, 4, figsize=(12,8), sharey=True, sharex=True)
            axs = axs.ravel()
            for i, ax in enumerate(axs):
                day = int(horizon*index[i])
                ax=plot_calibration(ax, pred[day], std_pred[day], true[day])
            fig.tight_layout()
            fig.savefig(f"../figures/{self.exp_name}/{hparams['encoder_type']}/{file_name}_calibration.pdf", dpi=480)
            plt.close()
            
        if trial is not None:
            return logger.metrics[-1]['val_mae_epoch']
        else:
            return metric

    def bactesting(self, baseline=False):
        """[Conduct backtesting cross-validation]

        Args:
            baseline (bool, optional): weather to run a baseline or FPSeQ model. Defaults to False.

        Returns:
            [type]: [description]
        """
        
        test_horizon=int(self.hparams['SAMPLES_PER_DAY']*31) 
        train_window_size=int(self.hparams['SAMPLES_PER_DAY']*31*6)
        step_size=int(self.hparams['SAMPLES_PER_DAY']*31)
        

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
            
            window = slice(experiment.data.index[train_index][0],  experiment.data.index[test_index][-1])
            train_window = slice(experiment.data.index[train_index][0], experiment.data.index[train_index][-1])
            test_window  = slice(experiment.data.index[test_index][0], experiment.data.index[test_index][-1])
            
            exp_name_full = f"cross_validation_{cross_validion}"
            self.train_window=train_window
            self.test_window=test_window
            print(f"---------------Backtesting expanding-{cross_validion} Cross validation Training --------------------------")
            print("")
            print(f"Train_window: {train_window}")
            print("")
            print(f"Test_window: {test_window}")
            print("")
            
            if baseline:
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
        fig.savefig(f"../figures/{self.exp_name}/MAAPE_{self.encoder_type}.pdf", dpi=480)
        plt.close()

        fig, ax = plt.subplots(1,1, figsize=(9,3))
        ax=plot_boxplot_per_encoder(ax, metrics_spilit, 'maap', 'MAAPE')
        ax.set_title(f"{self.encoder_type} {pd_metrics['maap'].mean():.{3}f} $\pm$ {pd_metrics['maap'].std():.{3}f}")
        plt.xticks(range(1, len(x_labels)+1), x_labels, rotation=0, fontsize=12);
        fig.savefig(f"../figures/{self.exp_name}/MAAPE_{self.encoder_type}.pdf", dpi=480)
        plt.close()

        if self.hparams['encoder_type'] not in ['LSTMBaseline', 'RF', 'SVR' ]:
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
            ax=plot_boxplot_per_encoder(ax, metrics_spilit, 'ciwrmse', 'CIWNMRSE')
            ax.set_title(f"{self.encoder_type} {pd_metrics['ciwrmse'].mean():.{3}f} $\pm$ {pd_metrics['ciwrmse'].std():.{3}f}")
            plt.xticks(range(1, len(x_labels)+1), x_labels, rotation=0, fontsize=12);
            fig.savefig(f"../figures/{self.exp_name}/CIWNMRSE_{self.encoder_type}.pdf", dpi=480)
            plt.close()

            


            fig, ax = plt.subplots(1,1, figsize=(9,3))
            ax=plot_boxplot_per_encoder(ax, metrics_spilit, 'train_time', 'TRA-TIME')
            ax.set_title(f"{self.encoder_type} {pd_metrics['train_time'].mean():.{3}f} $\pm$ {pd_metrics['train_time'].std():.{3}f}")
            plt.xticks(range(1, len(x_labels)+1), x_labels, rotation=0, fontsize=12);
            fig.savefig(f"../figures/{self.exp_name}/TRA-TIME_{self.encoder_type}.pdf", dpi=480)
            plt.close()

            fig, ax = plt.subplots(1,1, figsize=(9,3))
            ax=plot_boxplot_per_encoder(ax, metrics_spilit, 'test_time', 'TEST-TIME')
            ax.set_title(f"{self.encoder_type} {pd_metrics['test_time'].mean():.{3}f} $\pm$ {pd_metrics['test_time'].std():.{3}f}")
            plt.xticks(range(1, len(x_labels)+1), x_labels, rotation=0, fontsize=12);
            fig.savefig(f"../figures/{self.exp_name}/TEST-TIME_{self.encoder_type}.pdf", dpi=480)
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




def run_train_spilit(window=slice('2019-03', '2020-09'),
                    train_window=slice('2019-03', '2020-06'),
                    test_window=slice('2020-07', '2020-09'),
                    exp_name='train_test_spilit_samples_20210711_FPQMixerForecast', 
                    encoders = ['SVR','RF',  "MLPEncoder", "MLPMixerEncoder",  "GRUEncoder", "LSTMEncoder", "UNETEncoder"], 
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
    

    

    for encoder in encoders:
        for horizon in [48]:
            exp=Experiment(window=window,  
           train_window=train_window, test_window=test_window,
            exp_name=exp_name, seed = seed, encoder_type=encoder, sampling=sampling, train=train)
            if max_epochs is not None:
                exp.hparams.update({ 'max_epochs': max_epochs})

            if exp.hparams['seasonality']:
                file_name=f'{horizons[48]}_with_seasonality_forecasting'
            else:
                file_name=f'{horizons[48]}_forecasting'
    

            if encoder in ['SVR','RF' ]:
                metric = exp.baseline(file_name=file_name)
            else:
                metric = exp.train_spilit(file_name=file_name, horizon=horizon)
            
            #print(pd.DataFrame(metric.mean()).T)
            metrics_all[sampling]= metric
        pd_metrics_all[encoder]= metrics_all

    
    np.save(f"../results/{exp_name}/allmetrics.npy", pd_metrics_all)
        
            

    
def run_backtesting(window=slice('2019-03', '2020-09'), 
                    seed=777,
                    max_epochs=200,
                    train=True,
                    exp_name='backtesting_expanding_cross_validation_20210711', 
                    encoders = ["GRUEncoder", "MLPEncoder", "MLPMixerEncoder", "UNETEncoder", 'RF', 'SVR']):
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
    for encoder in encoders:
        #train=False if encoder in ['LSTMBaseline', 'GRUEncoder', "MLPEncoder", "MLPMixerEncoder", "UNETEncoder"] else True
        exp=Experiment(window=window, exp_name=exp_name, seed = seed, encoder_type=encoder, train=train)
        if max_epochs is not None:
            exp.hparams.update({ 'max_epochs': max_epochs})
        if encoder in ('RF', 'SVR'):
            baseline=True
        else:
            baseline=False
        pd_metrics, metrics_spilit = exp.bactesting(baseline=baseline)
        pd_metrics_all[encoder]= pd_metrics
        pd_metrics_spilit[encoder]=metrics_spilit
        print(f"{encoder}  {pd_metrics['maap'].mean():.{3}f} $\pm$ {pd_metrics['maap'].std():.{3}f}, \
        {pd_metrics['nrmse'].mean():.{3}f} $\pm$ {pd_metrics['nrmse'].std():.{3}f}, \
        {pd_metrics['corr'].mean():.{3}f} $\pm$ {pd_metrics['corr'].std():.{3}f}")

        if encoder not in ['RF', 'SVR' ]:
            print(f"{encoder}  {pd_metrics['maap'].mean():.{3}f} $\pm$ {pd_metrics['maap'].std():.{3}f}, \
            {pd_metrics['nrmse'].mean():.{3}f} $\pm$ {pd_metrics['nrmse'].std():.{3}f}, \
            {pd_metrics['pic'].mean():.{3}f} $\pm$ {pd_metrics['pic'].std():.{3}f}, \
            {pd_metrics['nmpi'].mean():.{3}f} $\pm$ {pd_metrics['nmpi'].std():.{3}f}, \
            {pd_metrics['ciwrmse'].mean():.{3}f} $\pm$ {pd_metrics['ciwrmse'].std():.{3}f} , \
            {pd_metrics['corr'].mean():.{3}f} $\pm$ {pd_metrics['corr'].std():.{3}f}")

    
    np.save(f"../results/{exp_name}/allmetrics.npy", pd_metrics_all)
    np.save(f"../results/{exp_name}/split_allmetrics.npy", pd_metrics_spilit)
            
    fig, ax = plt.subplots(1,1, figsize=(9,3))
    ax=plot_boxplot_per_encoder(ax, pd_metrics_all, 'nrmse', 'NRMSE')
    fig.savefig(f"../figures/{exp_name}/NRMSE.pdf", dpi=480)

        
    fig, ax = plt.subplots(1,1, figsize=(9,3))
    ax=plot_boxplot_per_encoder(ax, pd_metrics_all, 'maap', 'MAAPE')
    fig.savefig(f"../figures/{exp_name}/MAAPE.pdf", dpi=480)
        
    
    if encoder not in [ 'RF', 'SVR' ]:
        fig, ax = plt.subplots(1,1, figsize=(9,3))
        ax=plot_boxplot_per_encoder(ax, pd_metrics_all, 'pic', 'PICP')
        fig.savefig(f"../figures/{exp_name}/PICP.pdf", dpi=480)
        

        fig, ax = plt.subplots(1,1, figsize=(9,3))
        ax=plot_boxplot_per_encoder(ax, pd_metrics_all, 'nmpi', 'NMPI')
        fig.savefig(f"../figures/{exp_name}/NMPI.pdf", dpi=480)

        fig, ax = plt.subplots(1,1, figsize=(9,3))
        ax=plot_boxplot_per_encoder(ax, pd_metrics_all, 'ciwrmse', 'CIWNRMSE')
        fig.savefig(f"../figures/{exp_name}/CIWNRMSE.pdf", dpi=480)


        fig, ax = plt.subplots(1,1, figsize=(9,3))
        ax=plot_boxplot_per_encoder(ax, pd_metrics_all, 'train_time', 'TRA-TIME')
        fig.savefig(f"../figures/{exp_name}/TRA-TIME.pdf", dpi=480)
            
            
        fig, ax = plt.subplots(1,1, figsize=(9,3))
        ax=plot_boxplot_per_encoder(ax, pd_metrics_all, 'test_time', 'TEST-TIME')
        fig.savefig(f"../figures/{exp_name}/TEST-TIME.pdf", dpi=480)
            
            

       
if __name__ == "__main__":
    #run_train_spilit()
    #run_backtesting()
    #run_hyperparams()
    #run_real_time()

    
       
    
    
  
  
    