
from utils.metrics import get_daily_metrics, get_daily_pointwise_metrics
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import math
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.examples.bart import load_bart_od
from pyro.contrib.forecast import ForecastingModel, Forecaster, eval_crps
from pyro.infer.reparam import LinearHMMReparam, StableReparam, SymmetricStableReparam
from pyro.ops.tensor_utils import periodic_repeat
from pyro.ops.stats import quantile
from pyro.ops.tensor_utils import periodic_repeat
from pyro.ops.stats import quantile
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed 
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.seasonal import seasonal_decompose 
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from joblib import dump, load
from net.mdn_block import MDGMM, get_95_quantile_prediction, q
from timeit import default_timer
set_random_seed(0)
assert pyro.__version__.startswith('1.7.0')
pyro.set_rng_seed(20200305)




def save_model(model, file_name):
    dump(model, file_name) 







def load_model(file_name):
    return load(file_name) 



def make_pipeline(model, n_components=2):
    steps = list()
    steps.append(('pca', PCA(n_components=n_components)))
    steps.append(('model', model))
    pipeline = Pipeline(steps=steps)
    return pipeline

def get_prediction(X_train, X_test, pipeline, points=48):
    outputs = []

    for i in range(0, len(X_test)):
        data = X_train[-1] if i==0 else  X_test[i]
        for i in range (0, points):
            y_pred = pipeline.predict(np.array([data]))[0]
            outputs.append(y_pred)
            new_sample = data[1:]
            new_sample = np.append(new_sample, y_pred)
            data = new_sample
    pred=np.array(outputs).reshape(-1, points)
    return pred


def fit_baseline(train_loader, test_loader,experiment, index,  file_name, baseline='RF'):
    
    # fit the model
    if baseline=='RF':
        model=RandomForestRegressor()
    else:
        model=MultiOutputRegressor(SVR(kernel='rbf', C=1, gamma='auto', epsilon=0.01))
    pipeline = make_pipeline(model)
    X_train, y_train = train_loader.dataset.tensors

    start_time = default_timer() 
 
    pipeline.fit(X_train.numpy().reshape(len(X_train), -1), y_train.numpy())

    train_walltime = default_timer() - start_time
    
    X_test, y_test = test_loader.dataset.tensors
    start_time = default_timer()   
    y_hat = pipeline.predict(X_test.numpy().reshape(len(X_test), -1))
    test_walltime = default_timer() - start_time

    save_model(pipeline, f'{baseline}_{file_name}.joblib')
    pred, true = y_hat.reshape(-1, 1), y_test.reshape(-1, 1)

    pred = experiment.target_transformer.inverse_transform(pred)
    true = experiment.target_transformer.inverse_transform(true)
    target_range = true.max()-true.min()

    metrics=[]
    for i in range(0, len(pred), 48):
        metric= get_daily_pointwise_metrics(pred[i:i+48, 0].T, true[i:i+48, 0].T, target_range)
        metrics.append(metric)
        #print(index[i:i+48][0])
    metrics = pd.concat(metrics)

    print(pd.DataFrame(metrics.mean()).T[[ 'mae', 'nrmse' ,  'smape', 'corr' ]].round(2))
   
    metrics['train_time']=train_walltime
    metrics['test_time']=test_walltime
            
    logs  = {"pred": pred,  "true":true, 'target_range':target_range, 'metrics':metrics, 'index':index}
    
    return logs


class Model(ForecastingModel):
    def __init__(self, means):
        super().__init__()
        self.means = means
    def model(self, zero_data, covariates):
        duration = zero_data.size(-2)
        
        prediction = periodic_repeat(self.means, duration, dim=-1).unsqueeze(-1)

        # First sample the Gaussian-like parameters as in previous models.
        init_dist = dist.Normal(0, 10).expand([1]).to_event(1)
        timescale = pyro.sample("timescale", dist.LogNormal(math.log(48), 1))
        trans_matrix = torch.exp(-1 / timescale)[..., None, None]
        trans_scale = pyro.sample("trans_scale", dist.LogNormal(-0.5 * math.log(48), 1))
        obs_matrix = torch.tensor([[1.]])
        with pyro.plate("hour_of_week", 48 * 7, dim=-1):
            obs_scale = pyro.sample("obs_scale", dist.LogNormal(-2, 1))
        obs_scale = periodic_repeat(obs_scale, duration, dim=-1)

        # In addition to the Gaussian parameters, we will learn a global stability
        # parameter to determine tail weights, and an observation skew parameter.
        stability = pyro.sample("stability", dist.Uniform(1, 2).expand([1]).to_event(1))
        skew = pyro.sample("skew", dist.Uniform(-1, 1).expand([1]).to_event(1))

        # Next we construct stable distributions and a linear-stable HMM distribution.
        trans_dist = dist.Stable(stability, 0, trans_scale.unsqueeze(-1)).to_event(1)
        obs_dist = dist.Stable(stability, skew, obs_scale.unsqueeze(-1)).to_event(1)
        noise_dist = dist.LinearHMM(
            init_dist, trans_matrix, trans_dist, obs_matrix, obs_dist, duration=duration)

        # Finally we use a reparameterizer to enable inference.
        rep = LinearHMMReparam(None,                     # init_dist is already Gaussian.
                               SymmetricStableReparam(), # trans_dist is symmetric.
                               StableReparam())          # obs_dist is asymmetric.
        with poutine.reparam(config={"residual": rep}):
            self.predict(noise_dist, prediction)
            


def fit_space_state_forecasting(train_window, test_window, experiment, hparams):

    train_data = torch.from_numpy(experiment.data[hparams['targets']][train_window].values.astype(np.float64)).float().unsqueeze(1)
    test_data  = torch.from_numpy(experiment.data[hparams['targets']][test_window].values).float().unsqueeze(1)
    test_data = test_data[96:]
    
    index = experiment.data[test_window].iloc[96:].index
    T0=0
    T1=len(train_data)
    T2=T1+len(test_data)
    means = train_data[:T1 // (48 * 7) * 48 * 7].reshape(-1, 48* 7).mean(0)
    covariates = torch.arange(T0, float(T2)).unsqueeze(-1) / 356
    ghi = torch.from_numpy(pd.concat([experiment.data['Ghi'][train_window], experiment.data['Ghi'][test_window].iloc[96:]]).values)
    ts = torch.from_numpy(pd.concat([experiment.seasonalities[train_window], experiment.seasonalities[test_window].iloc[96:]]).values)
    covariates = torch.cat([ghi.unsqueeze(-1), ts], 1)
    
    
    pyro.set_rng_seed(1)
    pyro.clear_param_store()
    start_time = default_timer() 
    forecaster = Forecaster(Model(means), train_data, covariates[:T1],  learning_rate=0.1)
    for name, value in forecaster.guide.median().items():
        if value.numel() == 1:
            print("{} = {:0.4g}".format(name, value.item()))

    train_walltime = default_timer() - start_time
            
    
    start_time = default_timer()   
    samples = forecaster(train_data, covariates, num_samples=100)
    test_walltime = default_timer() - start_time


    pred,  quantiles , samples =get_95_quantile_prediction(gmm=None, samples=samples)
   
    

    
    q_pred = quantiles
    true = experiment.target_transformer.inverse_transform(test_data.numpy())
    pred = experiment.target_transformer.inverse_transform(pred.numpy().reshape(-1,1))
    q_pred = quantiles.numpy()
    N, M, T = q_pred.shape
    q_pred = q_pred.reshape(N*M, T)
    q_pred = experiment.target_transformer.inverse_transform(q_pred)
    q_pred = q_pred.reshape(N, M, T)
    target_range = true.max()-true.min()
    gt = experiment.target_transformer.inverse_transform(experiment.data[['Load']].values)
    #target_range=gt.max() - gt.min()
    tau_hats=np.array(q)
    tau_hats=torch.from_numpy(tau_hats[None, :, None]).expand_as(torch.from_numpy(q_pred)).numpy()

    M, N, T = samples.shape
    samples = samples.reshape(N*M, T)
    samples = experiment.target_transformer.inverse_transform(samples)
    samples = samples.reshape(M, N, T)
    
    #crps = eval_crps(torch.tensor(samples), torch.tensor(true))

    #_, pred, _= q_pred[:,0,:].squeeze(1), q_pred[:,len(q)//2,:], q_pred[:,-1,:].squeeze(1)
    #pred = samples.mean(0)
    metrics=[]
    for i in range(0, len(pred), 48):
        
        metric =get_daily_metrics(pred[i:i+48, 0].T, true[i:i+48, 0].T, q_pred[i:i+48,:, 0].T,
         target_range, 2*true[i:i+48, 0].T.std()/target_range, samples[:,i:i+48, 0], tau = tau_hats[i:i+48,:, 0].T,
                                    alpha=1.0)
        #metric=get_overall_metrics(true[i:i+48], pred[i:i+48], q_pred[i:i+48], tau_hats[i:i+48], crps, target_range)
        metrics.append(metric)
        #print(index[i:i+48][0])
    metrics = pd.concat(metrics)
    metrics['train_time']=train_walltime
    metrics['test_time']=test_walltime
    #metric=get_overall_metrics(true, pred, q_pred, tau_hats, crps, target_range)
    logs  = {"pred": pred,  "tau_hat":tau_hats, "q_pred":q_pred,  "true":true, 'target_range':target_range, 
             'metrics':metrics, 'samples':samples, 'index':index}
    print(pd.DataFrame(logs['metrics'].mean()).T[[  'mae', 'nrmse',  'ncrps',  'pic',  'nmpi', 'ciwe',  'ciwf', 'corr']].round(2))

    return logs



def fit_auto_regressive_forecasting(train_window, test_window, experiment, hparams):

    train_data = experiment.data[hparams['targets']][train_window]
    test_data  =experiment.data[hparams['targets']][test_window].iloc[96:]

    
    index = test_data.index
    
    train_data = train_data.reset_index()
    train_data = train_data.rename(columns={"timestamp": "ds", hparams['targets']: "y"})
    
    test_data = test_data.reset_index()
    test_data = test_data.rename(columns={"timestamp": "ds", hparams['targets']: "y"})
    
    T0=0
    T1=len(train_data)
    T2=T1+len(test_data)
    
    m = NeuralProphet(num_hidden_layers=2,d_hidden=64)

    start_time = default_timer() 
    metrics = m.fit(train_data, freq="30T")
    train_walltime = default_timer() - start_time
    
   
    start_time = default_timer()   
    future = m.make_future_dataframe(train_data,periods = len(test_data), n_historic_predictions = len(train_data))
    forecast = m.predict(future)
    test_walltime = default_timer() - start_time
    
   
    
    
    pred=forecast[['yhat1']][T1:T2].values
    true = experiment.target_transformer.inverse_transform(test_data[['y']].values)
    pred = experiment.target_transformer.inverse_transform(pred)
    target_range = true.max()-true.min()
    #gt = experiment.target_transformer.inverse_transform(experiment.data[['Load']].values)
    #target_range=gt.max() - gt.min()
    metrics=[]
    for i in range(0, len(pred), 48):
        metric=  get_daily_pointwise_metrics(pred[i:i+48, 0].T, true[i:i+48, 0].T, target_range)
        metrics.append(metric)
        #print(index[i:i+48][0])
    metrics = pd.concat(metrics)
    metrics['train_time']=train_walltime
    metrics['test_time']=test_walltime
   
    logs  = {"pred": pred,  "true":true, 'target_range':target_range, 
             'metrics':metrics, 'index':index}
    print(pd.DataFrame(logs['metrics'].mean()).T[[ 'mae', 'nrmse' ,  'smape', 'corr' ]].round(2))


    return logs



def fit_sarimax_forecasting(train_window, test_window,  experiment, hparams):

    train_data = experiment.data[hparams['targets']][train_window]
    test_data  =experiment.data[hparams['targets']][test_window].iloc[96:]

    
    index = test_data.index
    
    train_data = train_data.reset_index()
    train_data = train_data.rename(columns={"timestamp": "ds", hparams['targets']: "y"})
    
    test_data = test_data.reset_index()
    test_data = test_data.rename(columns={"timestamp": "ds", hparams['targets']: "y"})
    
    T0=0
    T1=len(train_data)
    T2=T1+len(test_data)
    
    
    sarima_order=[1,1,1]
    sarima_seasonal_order=[1,1,1,24]

    model = SARIMAX(endog=train_data['y'], order=sarima_order, seasonal_order=sarima_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)

    #Fits the model by maximum likelihood via Kalman filter and returns MLEResults class
    start_time = default_timer() 
    model_fit = model.fit(disp=False)
    train_walltime = default_timer() - start_time
    
    start_time = default_timer()   
    forecast_all = model_fit.get_forecast(len(test_data))
    test_walltime = default_timer() - start_time
    
    pred_mean = forecast_all.predicted_mean
    ci_95 = forecast_all.conf_int(alpha=0.05)
    ci_95['mean']=pred_mean.values
    q_pred=ci_95[['lower y', 'mean', 'upper y']].values[:,:,None]
    N, M, T = q_pred.shape
    q_pred = q_pred.reshape(N*M, T)
    q_pred = experiment.target_transformer.inverse_transform(q_pred)
    q_pred = q_pred.reshape(N, M, T)
    
    
    
    true = experiment.target_transformer.inverse_transform(test_data[['y']].values)
    target_range = true.max()-true.min()
    #gt = experiment.target_transformer.inverse_transform(experiment.data[['Load']].values)
    #target_range=gt.max() - gt.min()
    pred = q_pred[:,1,:]
    
    metrics=[]
    for i in range(0, len(pred), 48):
        metric=  get_daily_pointwise_metrics(pred[i:i+48, 0].T, true[i:i+48, 0].T, target_range)
        metrics.append(metric)
    metrics = pd.concat(metrics)
    metrics['train_time']=train_walltime
    metrics['test_time']=test_walltime
   
    
    




    logs  = {"metrics":metrics, "pred": pred,   "q_pred":q_pred,  "true":true, 'target_range':target_range, 
              'index':index}
    print(pd.DataFrame(logs['metrics'].mean()).T[[ 'mae', 'nrmse' ,  'smape', 'corr' ]].round(2))

    return logs
   
    

   
    