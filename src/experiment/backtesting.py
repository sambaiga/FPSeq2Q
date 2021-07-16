import numpy as np

def bactesting_sliding(duration, horizon, window, step_size, period=48):
    indices = np.arange(duration)
    start_index = 0
    anchor_index = start_index + window - period
    end_index = anchor_index + horizon
    
    while (end_index < duration):
        train_index=indices[start_index:anchor_index]
        test_index=indices[(anchor_index+period):end_index]
        
        print(f"Training  on window [{start_index}: {anchor_index}] of data, testing on window [{(anchor_index+1)}:{end_index}]")
        start_index = start_index + step_size
        anchor_index = start_index + window - period
        end_index = anchor_index + horizon
        
        yield train_index, test_index

def bactesting_sliding(duration, horizon, window, step_size, period=48):

    
    indices = np.arange(duration)
    start_index = 0
    anchor_index = start_index + window - period
    end_index = anchor_index + horizon
    
    while (end_index < duration):
        train_index=indices[start_index:anchor_index]
        test_index=indices[(anchor_index+period):end_index]
        
        print(f"Training  on window [{start_index}: {anchor_index}] of data, testing on window [{(anchor_index+1)}:{end_index}]")
        start_index = start_index + step_size
        anchor_index = start_index + window - period
        end_index = anchor_index + horizon
        
        yield train_index, test_index

def bactesting_expanding(duration, horizon, window, step_size, max_window, period=48):
    
    indices = np.arange(duration)
    start_index = 0
    anchor_index = start_index + window - period
    end_index = anchor_index + horizon
    
    while (anchor_index - start_index < max_window - period and end_index < duration):
        train_index=indices[start_index:anchor_index]
        test_index=indices[(anchor_index+period):end_index]
        
        print(f"Training  on window [{start_index}: {anchor_index}] of data, testing on window [{(anchor_index+1)}:{end_index}]")
        anchor_index = anchor_index + step_size 
        end_index = anchor_index + horizon  
        
        yield train_index, test_index
        
    if (end_index < duration):
        bactesting_sliding(duration, horizon, max_window, step_size)