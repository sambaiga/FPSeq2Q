import torch
import pytz
import pandas as pd
#from pytz import timezone
import numpy as np
import random
from numpy import array
import glob
import calendar
import time
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
    MinMaxScaler,
    FunctionTransformer
)

from torch.utils.data import TensorDataset, DataLoader
import scipy.signal as signal
from datetime import timedelta
from collections import OrderedDict
from datetime import datetime
from collections import OrderedDict
from dataclasses import dataclass, field
import logging
import inspect
import torch
import math
periods = ('15T', '30T', '60T')
samples = (96, 48, 24)
log = logging.getLogger("NL")



def fourier_series(dates, period, series_order):
    """Provides Fourier series components with the specified frequency and order.
    Note: Identical to OG Prophet.
    Args:
        dates (pd.Series): containing timestamps.
        period (float): Number of days of the period.
        series_order (int): Number of fourier components.
    Returns:
        Matrix with seasonality features.
    """
    # convert to days since epoch
    t = np.array((dates - datetime(1970, 1, 1)).dt.total_seconds().astype(float)) / (3600 * 24.0)
    return fourier_series_t(t, period, series_order)


def fourier_series_t(t, period, series_order):
    """Provides Fourier series components with the specified frequency and order.
    Note: Identical to OG Prophet.
    Args:
        t (pd.Series, float): containing time as floating point number of days.
        period (float): Number of days of the period.
        series_order (int): Number of fourier components.
    Returns:
        Matrix with seasonality features.
    """
    features = np.column_stack(
        [fun((2.0 * (i + 1) * np.pi * t / period)) for i in range(series_order) for fun in (np.sin, np.cos)]
    )
    return features





def loadData(a_path, a_cols, a_rename = None, a_idx_field="timestamp", a_period="1T", a_timezone = None):
    data = pd.DataFrame()
    for path in a_path:
        data_temp = pd.read_csv(path, usecols = a_cols)
        data  = pd.concat([data, data_temp], ignore_index=True)
    
    
    data[a_idx_field] = pd.to_datetime(data[a_idx_field], utc=True)

    
    if a_rename != None:
        data.rename(columns = a_rename, inplace=True)
    data.set_index(a_idx_field, inplace=True)
    #data = data.interpolate(method="time")
    if a_timezone != None:
        data.index = data.index.tz_convert(a_timezone)
        
    data.index = data.index.astype('datetime64[ns]')
    data = data.resample(a_period).mean()
    #data = resample(data, rate=a_period, short_rate='T', max_gap="30T")
    
    return data


def load_dataset(a_period="30T"):
    """
    Load an already cleaned version of the dataset
    """
    
    
    load = loadData(['../data/Substation_2019_to_16042021.csv'], 
                a_cols = ["timestamp", "measure_cons"], a_rename={"measure_cons":"Load"}, a_period=a_period, 
                a_timezone = "Atlantic/Madeira")

    radiation = loadData(['../data/Solcast_PT5M_2019.csv','../data/Solcast_PT5M_2020.csv'], 
               a_cols = ["PeriodEnd", "Ghi"], a_idx_field="PeriodEnd", a_period=a_period, 
               a_timezone = "Atlantic/Madeira")


    #radiation2 = loadData(['../data/Radiation-historical_30min.csv'], 
    #           a_cols = ["PeriodEnd", "Ghi", "Dhi", "CloudOpacity"], a_idx_field="PeriodEnd", a_period=a_period, 
    #           a_timezone = "Atlantic/Madeira")


    #radiation_cop = loadData(['../data/Fazendinha_cams-solar-radiation-timeseries_2020_15minute.csv'],
    #                     a_cols = ["timestamp", "GHI"], a_rename = {"GHI":"GhiC"}, a_period="30T")

    # add the remaining data of 2020
    #rad = pd.merge(radiation, radiation_cop, how = 'outer', left_index = True, right_index = True)
    
    #rad = pd.concat([radiation, radiation_cop], join="inner")

    return load, radiation

def add_exogenous_variables(df, one_hot=True):
    """
    Augument the dataframe with exogenous features (date/time feature + holiday feature).
    The feature's values can be kept as they are or they can be one hot encoded
    :param df: the dataframe
    :param one_hot: if True, one hot encode all the features.
    :return: two matrix of exogenous features,
     the first for temperatures only the second one contains all the other variables.
    """
    data = df.copy().reset_index()
    data['DAYOFWEEK'] = data.timestamp.dt.dayofweek
    data['WEEK'] = data.timestamp.dt.isocalendar().week if hasattr(data.timestamp.dt, "isocalendar") else data.timestamp.dt.week
    data['DAYOFYEAR'] = data.timestamp.dt.dayofyear
    data['MONTH'] = data.timestamp.dt.month
    data['DAY'] = data.timestamp.dt.day
    data['HOUR'] = data.timestamp.dt.hour

    data["WEEKDAY"] = 0
    data['WEEKDAY'] = ((data.timestamp.dt.dayofweek) // 5 == 0).astype(float)
    data["WEEKEND"] = 0
    data["WEEKEND"] = ((data.timestamp.dt.dayofweek) // 5 == 1).astype(float)

    data["SATURDAY"] = 0
    data['SATURDAY'] = ((data.timestamp.dt.dayofweek == 5)).astype(float)

    data["SUNDAY"] = 0
    data['SUNDAY'] = ((data.timestamp.dt.dayofweek == 6)).astype(float)


   
    if one_hot:
        ex_feat = pd.get_dummies(data, columns=['MONTH', 'DAY', 'HOUR', 'DAYOFWEEK', 'DAYOFYEAR', "WEEKDAY","WEEKEND", "SATURDAY", "SUNDAY" ])
        return ex_feat
    else:
        return data


def get_dataset(period  = '30T', rolling_window=3, SAMPLES_PER_DAY=48,
                combine=False, add_ghi_feature=True,  cleanData=True,
                window=slice('2019-01-01', '2019-12-31')):
    
    load, radiation = load_dataset(a_period=period)

    #create datefeatures
    load=add_exogenous_variables(load, one_hot=False)
    load = load.set_index("timestamp")
    load = load.loc[window]
    radiation=radiation.loc[window]


    print(f"Total data sample: {load.shape[0]}")
    print(f"Missing data sample: {load.isnull().sum()[0]}")
    print(f" percentage of Missing data sample: {load.isnull().sum()[0]/len(load)}")

    if combine:
        print("merge")
        data = pd.merge(radiation, load, how = 'outer', left_index = True, right_index = True)

        #create other added feature by comnining loads and radition
    else:
        data = load

    #remove some gap that span the whole day
    if cleanData == True:
        d = data.groupby(pd.Grouper(freq='D')).count()
        d = d[d.Load < SAMPLES_PER_DAY]
        dates = list(d.index.strftime("%Y-%m-%d"))

        index_names = data[data.index.strftime("%Y-%m-%d").isin(dates)].index

        print("Before Clean", data.shape)
        data.drop(index_names, inplace = True) 
        print("After Clean", data.shape)

    data.sort_index(inplace=True)
    data.dropna(inplace = True)
    print(" ")
    print(f"Total data sample after cleaning: {data.shape[0]}")
    print(f"Missing data sample after cleaning: {data.isnull().sum()[0]}")
    print(f" percentage of data loss : {1- (data.shape[0]/len(load))}")

    
    data['Load_median']= data['Load'].rolling(rolling_window).median().fillna(method='bfill').fillna(method='ffill')
    data['Load_mean']= data['Load'].rolling(rolling_window).mean().fillna(method='bfill').fillna(method='ffill')
    data['Load_std'] = data['Load'].rolling(rolling_window).std().fillna(method='bfill').fillna(method='ffill')
    data['Load-median-filter'] = signal.medfilt(data['Load'].values.flatten(), kernel_size=rolling_window)
    

   
    return data





class TimeSeriesDataset(object):   
    def __init__(self, unknown_features, kown_features, targets,categorical_features=None, window_size=96, horizon=48, batch_size=64, shuffle=False):
        self.inputs = unknown_features
        self.covariates = kown_features
        self.categorical = categorical_features
        self.targets = targets
        self.window_size = window_size
        self.horizon = horizon
        self.batch_size = batch_size
        self.shuffle = shuffle

    def frame_series(self):
        
        nb_obs, nb_features = self.inputs.shape
        features, targets, covariates = [], [], []
        if self.categorical is not None:
            categorical, categorical_covariates = [], []

        for i in range(0, nb_obs - self.window_size - self.horizon+1):
            features.append(torch.FloatTensor(self.inputs[i:i + self.window_size, :]).unsqueeze(0))
            targets.append(
                    torch.FloatTensor(self.targets[i + self.window_size:i + self.window_size + self.horizon]).unsqueeze(0))
            covariates.append(
                    torch.FloatTensor(self.covariates[i + self.window_size:i + self.window_size + self.horizon,:]).unsqueeze(0))

            
            if self.categorical is not None:
                categorical.append(torch.LongTensor(self.categorical[i:i + self.window_size, :]).unsqueeze(0))
                categorical_covariates.append(torch.LongTensor(self.categorical[i + self.window_size:i + self.window_size + self.horizon]).unsqueeze(0))
            
            
        features = torch.cat(features)
        targets, covariates = torch.cat(targets), torch.cat(covariates)
        
        
        
        #padd covariate features with zero
        diff = features.shape[2] - covariates.shape[2]
        covariates = torch.nn.functional.pad(covariates, [diff// 2, diff - diff // 2])
        features = torch.cat([features, covariates], dim=1)
        del covariates
        
        if self.categorical is not None:
            categorical, categorical_covariates = torch.cat(categorical), torch.cat(categorical_covariates)
            categorical = torch.cat([categorical, categorical_covariates], 1)
            del categorical_covariates
            
            return TensorDataset(features,  categorical, targets)
        else:

            return TensorDataset(features,  targets)
    
    def get_loader(self):
        dataset = self.frame_series()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=True, num_workers=4, pin_memory=True)
        return loader
        



class DatasetObjective(object):

    def __init__(self, hparams,  combine=True, add_ghi_feature=True,  cleanData=False,
                window=slice('2019-01-01', '2019-12-31'), data=None):



        hparams.update({'period': periods[hparams['sampling']]})
        hparams.update({'SAMPLES_PER_DAY': samples[hparams['sampling']]})
        hparams.update({ 'window_size': 2*samples[hparams['sampling']]})
        hparams.update({ 'horizon': samples[hparams['sampling']]})
        self.hparams = hparams
        print(f"load data for {window} window")
        if data is None:
            if isinstance(window, list):
                data = []
                for  w in window:
                    d =  get_dataset(period  = hparams['period'], 
                                    rolling_window=hparams['rolling_window'], 
                                    SAMPLES_PER_DAY=hparams['SAMPLES_PER_DAY'],
                                    combine=combine, add_ghi_feature=add_ghi_feature,  
                                    cleanData=cleanData,
                                    window=w)
                    
                    data.append(d)
                self.data = pd.concat(data)

                    

            else:
                self.data =  get_dataset(period  = hparams['period'], 
                                    rolling_window=hparams['rolling_window'], 
                                    SAMPLES_PER_DAY=hparams['SAMPLES_PER_DAY'],
                                    combine=combine, add_ghi_feature=add_ghi_feature,  
                                    cleanData=cleanData,
                                    window=window)

        else:
            self.data = data

        self.categorical_scaler = MinMaxScaler()
        self.seasonal_scaler = MinMaxScaler()
        self.ghi_scaler  = MinMaxScaler()
        self.load_scaler = MinMaxScaler()
        if hparams['scale']=='min_max':
             self.scaler = MinMaxScaler()
        elif hparams['scale']=='standard_scaler':
             self.scaler = StandardScaler()
        elif hparams['scale']=='quantile_scaler':
             self.scaler = QuantileTransformer(n_quantiles=hparams['N'])
        elif hparams['scale'] =='power_scaler':
            self.scaler = PowerTransformer()
        
        if hparams['target_scale']=='min_max':
            self.target_transformer = MinMaxScaler(feature_range=(-1, 1))

        if hparams['target_scale']=='standard_scaler':
            self.target_transformer = StandardScaler()
        elif hparams['target_scale']=='power_scaler':
            self.target_transformer = PowerTransformer()

        elif hparams['target_scale']=='quantile_scaler':
            self.target_transformer =QuantileTransformer(n_quantiles=hparams['N'])
        elif hparams['target_scale']=='log_scaler':
            self.target_transformer = FunctionTransformer(np.log1p)
            self.data[hparams['targets']]  = np.where(self.data[hparams['targets']]<0, 0.0, self.data[hparams['targets']])

        
        #compute load-ghi
        load=self.load_scaler.fit_transform(self.data[['Load-median-filter']].values)
        ghi=self.ghi_scaler.fit_transform(self.data[['Ghi']].values)
        
        
        print('Compute load ghi feature')
        Load_ghi=[]
        for i in range(hparams['SAMPLES_PER_DAY'], len(self.data)-hparams['SAMPLES_PER_DAY'], hparams['SAMPLES_PER_DAY']):
            lg=load[i-hparams['SAMPLES_PER_DAY']:i]- ghi[i:i+hparams['SAMPLES_PER_DAY']]
            Load_ghi.append(lg)
        Load_ghi = np.vstack(Load_ghi)
        
        self.data = self.data.iloc[:len(Load_ghi)]
        self.data['Load-Ghi']=Load_ghi
        self.target = self.data[['Load-median-filter']].values

       

        self.numerical_features = hparams['time_varying_unknown_feature'] + hparams['time_varying_known_feature']
        
        if not hparams['categorical_emb']:
            self.data[hparams['time_varying_known_categorical_feature']] = self.categorical_scaler.fit_transform(self.data[hparams['time_varying_known_categorical_feature']])
        else:
            self.data[hparams['time_varying_known_categorical_feature']] = self.data[hparams['time_varying_known_categorical_feature']].astype(np.float32).values

        
        
        self.target_transformer.fit(self.target)
        self.data[self.numerical_features] = self.scaler.fit_transform(self.data[self.numerical_features])
        self.data[hparams['targets']] = self.target_transformer.transform(self.target) 


        
        if self.data.index.name!='timestamp':
            self.data.index.name='timestamp'
        exog = self.categorical_scaler.inverse_transform(self.data[hparams['time_varying_known_categorical_feature']])
        exog_periods=[len(np.unique(exog[:, l])) for l in range(exog.shape[-1])]
        self.exog_periods = exog_periods
        seasonalities =np.hstack([fourier_series_t(exog[:,i], exog_periods[i], 1) for i in range(len(exog_periods))])
        seasonalities = self.seasonal_scaler.fit_transform(seasonalities)
        #seasonalities = seasonal_features_from_dates(self.data.reset_index()["timestamp"], self.season_config)
        self.seasonalities=pd.DataFrame(seasonalities)
        self.seasonalities.index = self.data.index
        
        self.seasonality_columns= [f"{i+1}" for i in range(seasonalities.shape[-1])]
        for i, column in enumerate(self.seasonality_columns):
            self.data[column]=seasonalities[:, i]
        



    def get_dataset(self, hparams, window= slice('2020-03', '2020-06'), shufle=True):
        
        data = self.data.loc[window]
        numerical_features = hparams['time_varying_unknown_feature'] + hparams['time_varying_known_feature']
        if not self.hparams['seasonality']:
            known_features = data[hparams['time_varying_known_feature'] + hparams['time_varying_known_categorical_feature']].values.astype(np.float64)
            unkown_features = data[numerical_features + hparams['time_varying_known_categorical_feature']].values.astype(np.float64)
        
        if self.hparams['seasonality']:
            seasonalities=data[self.seasonality_columns].values
            known_features = np.concatenate([data[hparams['time_varying_known_feature']].values, seasonalities], 1).astype(np.float64)
            unkown_features = np.concatenate([data[numerical_features],seasonalities], 1).astype(np.float64)
            
        target = data[hparams['targets']].values.astype(np.float64)
        
        loader = TimeSeriesDataset(unkown_features, known_features, 
                            target, None,
                            hparams['window_size'], 
                            hparams['horizon'], 
                            hparams['batch_size'], 
                            shuffle=shufle).get_loader()
    
        return loader