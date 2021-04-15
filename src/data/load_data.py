import pandas as pd
import numpy as np

appliance_names = {'kettle':'KT', 'fridge':'FRZ', 'dishwasher':'DW', 
                   'washingmachine':"WM", 'microwave':"MW", 'washer dryer':"WD", 
                   'television':"TV", 'boiler':"BL"}
ukdale_appliance_data = {
    "kettle": {
        "mean": 700,
        "std": 1000,
        'window':5,
        'on_power_threshold': 20,
        'max_on_power': 3998
    },
    "fridge": {
        "mean": 200,
        "std": 400,
        "window":50,
        'on_power_threshold': 50,
        
      
    },
    "dishwasher": {
        "mean": 700,
        "std": 700,
        "window":120,
        'on_power_threshold': 10
    },
    
    "washingmachine": {
        "mean": 400,
        "std": 700,
        "window":150,
        'on_power_threshold': 20,
        'max_on_power': 3999
    },
    "microwave": {
        "mean": 500,
        "std": 800,
        "window":5,
        'on_power_threshold': 20,
       
    },
    "boiler": {
        "mean": 26.5,
        "std": 24.12,
        "window":50,
        'on_power_threshold': 20,
       
    },
}






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
    
def get_percentile(data,p=50):
    """[summary]
    
    Arguments:
        data {[type]} -- [description]
        quantile {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    return np.percentile(data, p, axis=1, interpolation="nearest")

