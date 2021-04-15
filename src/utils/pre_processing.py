import numpy as np
import torch
import pandas as pd
from scipy import stats
import arviz as az
import matplotlib.pyplot as plt
az.style.use(["science", "high-vis"])

appliances=['kettle', 'fridge', 'dish washer', 'washing machine', 'microwave',  'television']
appliance_metadata={
    'on_threshold': 
    {'washer dryer':50, 'dish washer':10, 'television':50, 'kettle':2000, 'microwave':200, "toaster":10,
     'boiler':25, 'fridge freezer':50, 'cooker':10, 'washing machine':20, 'fridge':50, 'vacuum cleaner':20},
    'on_duration': 
    {'washer dryer':120, 'dish washer':120, 'television':50, "toaster":10,
     'boiler':10, 'fridge freezer':50, 'kettle':10, 'microwave':10, 'cooker':10, 'washing machine':150, 'fridge':10}
    }
appliance_names = {'kettle':'KT', 'fridge':'FRZ', 'dish washer':'DW', 
                   'washing machine':"WM", 'microwave':"MW", 'washer dryer':"WD", 
                   'television':"TV"}

def spilit_refit_test(data):
    split_1 = int(0.60 * len(data))
    split_2 = int(0.85 * len(data))
    train = data[:split_1]
    validation = data[split_1:split_2]
    test = data[split_2:]
    return train, validation, test

def binarization(data,threshold):
    """[summary]
    
    Arguments:
        data {[type]} -- [description]
        threshold {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    state = np.where(data>= threshold,1,0).astype(int)
    return state
    


def get_differential_power(data):
    """
    Generate differential power:=p[t+1]-p[t]
    :arg
    data:  power signal
    :return
    differential_power
    """
    return np.ediff1d(data, to_end=None, to_begin=0)   

 
def get_variant_power(data, alpha=0.1):
    """
    Generate variant power which reduce noise that may impose negative inluence on  pattern identification
    :arg
    data:  power signal
    alpha[0,0.9]:  reflection rate
    :return
    variant_power: The variant power generated
    """
    variant_power = np.zeros(len(data))
    for i in range(1,len(data)):
            d = data[i]-variant_power[i-1]
            variant_power[i] = variant_power[i-1] + alpha*d     
    return  variant_power 

    

def get_percentile(data,p=50):
    """[summary]
    
    Arguments:
        data {[type]} -- [description]
        quantile {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    return np.percentile(data, p, axis=1, interpolation="midpoint")


def over_lapping_sliding_window(data, seq_len = 4, step_size = 1):
    """over_lapping_sliding_window

    Args:
        data ([type]): [description]
        seq_len (int, optional): [description]. Defaults to 4.
        step_size (int, optional): [description]. Defaults to 1.

    Returns:
        [array]: [sequence]
    """
    units_to_pad = (seq_len//2)*step_size
    new_data = np.pad(data, (units_to_pad,units_to_pad),'constant',constant_values=(0,0))
    sh = (new_data.size - seq_len, seq_len)
    st = new_data.strides * 2
    sequences = np.lib.stride_tricks.as_strided(new_data, strides = st, shape = sh)[0::step_size]
    return sequences.copy()



def quantile_filter(data:np.array, sequence_length:int=10,  p:int=50):
    """ quantile_filter

    Args:
        sequence_length ([type]): [description]
        data ([array]): [description]
        p (int, optional): [description]. Defaults to 50.

    Returns:
        [type]: [description]
    """
    new_mains = over_lapping_sliding_window(data, sequence_length)
    new_mains = get_percentile(new_mains, p)
    return new_mains


def get_differential_power(data):
    return np.ediff1d(data, to_end=None, to_begin=0)   
    


def get_variant_power(data, alpha=0.1):
    """
    Generate variant power which reduce noise that may impose negative inluence on  pattern identification
    :arg
    data:  power signal
    alpha[0,0.9]:  reflection rate
    :return
    variant_power: The variant power generated
    """
    variant_power = np.zeros(len(data))
    for i in range(1,len(data)):
            d = data[i]-variant_power[i-1]
            variant_power[i] = variant_power[i-1] + alpha*d     
    return  variant_power 



def get_temporal_info(data):
    """generate extra information about the temporal information related 
    power consumption
    Args:
        data (list(DatetimeIndex)): a list of all temporal information
    Returns:
        [type]: [description]
    """
    
    out_info =[]
    for d in data:
        dow = d.dt.dayofweek.values / 7
        doy = d.dt.day.values / 360
        hod = d.dt.hour.values / 24
        woy = d.dt.week.values / 52
        minutes = d.dt.minute.values / 60
        seconds = (d - d.iloc[0]).dt.total_seconds() / 86400 
        out_info.append([seconds, minutes, hod, dow, doy, woy ])
    
    
    return  np.transpose(np.array(out_info)).reshape((-1,6))


def data_preprocessing(data_path="../data/ukdale/ukdale_jan_march.csv", appliance=None, 
                       feature_type="all",
                       alpha=0.05,
                       normalize=None, 
                       main_mu=233, 
                       main_std=494,
                      q_filter={"q":50, "w":20}, vis=True, denoise=True):


    data = pd.read_csv(f"{data_path}")
    sub_mains = data[appliance].sum(axis=1).values.flatten()
    main =  data['sub_mains'].values.flatten() if denoise else data['mains'].values.flatten()
    main = np.where(main<sub_mains, sub_mains, main)
    res = main-sub_mains

    if q_filter is not None:
        main = quantile_filter(data=main , sequence_length=q_filter['w'],  p=q_filter['q'])
        res = quantile_filter(data=res , sequence_length=q_filter['w'],  p=q_filter['q'])
        
    
    if vis:
        plt.figure(figsize=(9,3))
        #plt.plot(sub_mains[40000:45000])
        plt.plot(main[40000:45000])
        plt.plot(res[40000:45000])
        plt.show()

    
    targets = []
    states = [] 
    
    if vis:
        plt.figure(figsize=(9, 5))
        plt.subplot(4,2,1)
        plt.plot(main[40000:45000])
        plt.ylabel("Power $(W)$")
        ax = plt.gca()
        ax.autoscale(tight=True)
        plt.tight_layout() 
    for app_idx, app in enumerate(appliance):
        power = data[app].values
        meter = quantile_filter(data=power , sequence_length=appliance_metadata['on_duration'][app],  p=q_filter['q'])
        state = binarization(meter,appliance_metadata['on_threshold'][app])
        if q_filter is not None:
            power = quantile_filter(data=power , sequence_length=q_filter['w'],  p=q_filter['q'])
        targets.append(power)
        states.append(state)
        if vis:
            plt.subplot(4,2,app_idx+2)
            plt.title(app)
            plt.plot(power[40000:45000])
            plt.ylabel("Power $(W)$")

            ax = plt.gca()
            ax.autoscale(tight=True)
            plt.tight_layout() 
    
    
    
        
    states = np.stack(states).T
    targets = np.stack(targets).T
    targets = np.log1p(targets)
    
    #normalize inputs
    #main = stats.zscore(main)
    
    
    
    
    if feature_type=="diff":
        main = get_differential_power(main)[:,None]
          
    elif feature_type=="vpower" and alpha>0:
            
        dv = get_variant_power(main, alpha)[:,None]
        main = get_differential_power(dv.flatten())[:,None]
           
            
    elif feature_type=="combined" and alpha>0:
        dp = get_differential_power(main)[:,None]
        dv = get_variant_power(main, alpha)[:,None]
        dvp = get_differential_power(dv.flatten())[:,None]
        main = np.concatenate([main[:,None], dv, dp, dvp], axis=1)
        if vis:
            plt.figure(figsize=(9, 5))
            for k in range(main.shape[1]):
                plt.subplot(2,2,k+1)
                plt.plot(main[40000:45000, k])
                plt.ylabel("Power $(W)$")
                ax = plt.gca()
                ax.autoscale(tight=True)
                plt.tight_layout() 
                
    elif feature_type=="combined_time" and alpha>0:
        dp = get_differential_power(main)[:,None]
        dv = get_variant_power(main, alpha)[:,None]
        dvp = get_differential_power(dv.flatten())[:,None]
        
        main = np.concatenate([main[:,None],  data[['dayofweek']].values, data[['hour']].values, data[['minute']].values ], axis=1)
        
    

        if vis:
            plt.figure(figsize=(9, 5))
            for k in range(main.shape[1]):
                plt.subplot(2,2,k+1)
                plt.plot(main[40000:45000, k])
                plt.ylabel("Power $(W)$")
                ax = plt.gca()
                ax.autoscale(tight=True)
                plt.tight_layout() 
    
        
    return main, targets, states
   

class Dataset(torch.utils.data.Dataset):
    def __init__(self,  inputs, targets, states,  seq_len=99):
        self.inputs = inputs
        self.targets = targets
        self.states  = states
        seq_len = seq_len  if seq_len% 2==0 else seq_len+1
        self.seq_len = seq_len
        self.len = self.inputs.shape[0] - self.seq_len
        self.indices = np.arange(self.inputs.shape[0])
    def __len__(self):
        'Denotes the total number of samples'
        return self.len
    
    def get_sample(self, index):
        indices = self.indices[index : index + self.seq_len]
        inds_inputs=sorted(indices[:self.seq_len])
        inds_targs=sorted(indices[self.seq_len-1:self.seq_len])

        return self.inputs[inds_inputs], self.targets[inds_targs], self.states[inds_targs]

    def __getitem__(self, index):
        inputs, target, state = self.get_sample(index)
        return torch.tensor(inputs).float(), torch.tensor(target).float().squeeze(), torch.tensor(state).long().squeeze()
    