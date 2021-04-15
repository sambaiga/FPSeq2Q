
import torch
import numpy as np
import random
import arviz as az
import pandas  as pd
import matplotlib.pyplot as plt
az.style.use(["science", "grid"])
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from .load_data import get_percentile, binarization, ukdale_appliance_data, appliance_names
from utils.visual_functions import get_label_distribution
import warnings
warnings.filterwarnings("ignore")

def get_data(data_path="../data/ukdale/ukdale_jan_june.csv"):
    data = pd.read_csv(data_path)
    columns = {'boiler':'boiler', 'fridge':'fridge', 'washing machine':'washingmachine', 'dish washer':'dishwasher', 'kettle':'kettle', 'microwave':'microwave'}
    data.rename(columns, axis=1, inplace=True)
    data.set_index("Unnamed: 0", drop=True, inplace=True)
    data.index = pd.to_datetime(data.index)

    return data


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

def process_data(data, vis=False):
    
    targets = []
    states = [] 

    for app in list(ukdale_appliance_data.keys()):
        power = data[app].values
        meter =  quantile_filter(data=power , sequence_length=ukdale_appliance_data[app]['window'],  p=50)
        state = binarization(meter,ukdale_appliance_data[app]['on_power_threshold'])
        meter = (power - ukdale_appliance_data[app]['mean'])/ukdale_appliance_data[app]['std']
        meter = quantile_filter(data=meter , sequence_length=5,  p=50)
        #
        
        
        targets.append(meter)
        states.append(state)
    #states = np.stack(states).T
    #targets = np.stack(targets).T
    mains = data.sub_mains.values.flatten()
    mains = data[list(ukdale_appliance_data.keys())].sum(1).values.flatten()
    mains = quantile_filter(data=mains , sequence_length=5,  p=50)
    mains = (mains - mains.mean())/mains.std()
    
    size=min([len(s) for s in states])
    mains=mains[:size]
    states = np.stack([s[:size] for s in states]).T
    targets = np.stack([s[:size] for s in targets]).T
    
    
    if vis:
        fig = plt.figure(figsize=(6, 3))
        for i, app in enumerate(list(ukdale_appliance_data.keys())):
            
            ax  = fig.add_subplot(2,3,i+1)
            ax  = get_label_distribution(ax, states[:,i], title=appliance_names[app])
            ax = plt.gca()
            ax.autoscale(tight=True)
            plt.tight_layout() 
        fig.tight_layout()
    
    return mains, targets, states


def train_test_split(data, vis=False):
    
    mskf = MultilabelStratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    #train and validate on first four months (Jan, Feb, March, and April)
    print("Prepare training set")
    x, y,z=process_data(data["2015-01":"2015-05"], vis)
    train_index, test_index  =  next(mskf.split(x, z))
    
    x_train, x_val = x[train_index], x[test_index]
    y_train, y_val = y[train_index], y[test_index]
    z_train, z_val = z[train_index], z[test_index]
    print("Prepare test set")
    #test on last two weeks
    x_test, y_test,z_test=process_data(data["2015-06-15":"2015-06-30"], vis)
    
    print(f"Train:{round(100*len(x_train)/len(data), 0)} Val:{round(100*len(x_val)/len(data),0 )} Test:{round(100*len(x_test)/len(data),0 )}")
    
    return dict(x_test=x_test, y_test=y_test, z_test=z_test,
                              x_val=x_val, y_val=y_val, z_val=z_val,
                              x_train=x_train, y_train=y_train, z_train=z_train)


class DataGenerator(object):
    
    def __init__(self,  inputs, targets, labels, 
                 seq_length=50,  batchsize=32, 
                 shuffle=True, ignore_last_batch=True, 
                 filter_prob=0.5):
        if inputs.ndim==1:
            inputs = inputs[:, None] 
        self.batchsize = batchsize
        self.seq_length = seq_length
        self.filter_prob = filter_prob
        self.shuffle  = shuffle
        self.ignore_last_batch = ignore_last_batch
        
        self.inputs  = torch.tensor(inputs).float()
        self.targets = torch.tensor(targets).float()
        self.labels = torch.tensor(labels).float()
        self.i = 0
        self.iter = 0
        self.len = len(self.inputs) - seq_length

        self.indices = np.arange(self.len) 
        if self.ignore_last_batch:
            self.num_batches = (self.len // batchsize) -1
        else:
            self.num_batches = self.len // batchsize

        self.new_epoch()
       
        
        print("Input data: {} targets: {} batches: {}".format(self.inputs.shape, self.targets.shape, self.num_batches))



    def new_epoch(self):
        self.iter = 0 
        
        if self.shuffle:
            if self.filter_prob:
                self.filter_data()
            self.shuffle_data()
            
    def filter_data(self):
        prob = np.random.uniform(low=0.0, high=1.0, size=None)
        idxNonZero=~np.all(self.labels.numpy() == 0, axis=1)
        if prob < self.filter_prob:
            self.targets = self.targets[idxNonZero]
            self.inputs  = self.inputs[idxNonZero]
            self.labels  = self.labels[idxNonZero]
            self.len = len(self.inputs) - self.seq_length
            if self.ignore_last_batch:
                self.num_batches = (self.len // self.batchsize) -1
            else:
                self.num_batches = self.len // self.batchsize
            self.indices=np.arange(self.len)

    def shuffle_data(self):
        new_order = np.random.permutation(self.len)
        self.indices = self.indices[new_order]
        
        

    def __iter__(self):
        self.i,self.iter = 0,0
        if self.shuffle:
            self.shuffle_data()
        return self

    def __len__(self): 
        return self.num_batches

    def __next__(self):
        
        if self.iter > self.num_batches:
            self.new_epoch()
            raise StopIteration()
        
    
        out = self.get_batch(self.i)
        self.i += self.batchsize
        self.iter += 1
        return out


    
    def get_batch(self, start_idx):
        
        excerpt = self.indices[start_idx:start_idx + self.batchsize]
        inputs=torch.stack([self.inputs[idx:idx + self.seq_length] for idx in excerpt])
        targets = self.targets[excerpt]
        labels = self.labels[excerpt]

    
        return inputs, targets, labels