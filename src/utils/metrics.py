import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pyro.contrib.forecast import  eval_crps
import torch
import pandas as pd
from scipy import stats


def get_pointwise_metrics(pred:np.array, true:np.array, target_range:float):
    """calculate pointwise metrics
    Args:   pred: predicted values
            true: true values
            target_range: target range          
    Returns:    rmse: root mean square error                


    """
    assert pred.ndim == 1, "pred must be 1-dimensional"
    assert true.ndim == 1, "pred must be 1-dimensional"
    assert pred.shape == true.shape, "pred and true must have the same shape"
    #target_range = true.max() - true.min()
    

    nrmse =min( np.sqrt(mean_squared_error(true, pred))/target_range, 1)
    mae = mean_absolute_error(true, pred)/1000
    smape = 2 * np.mean(np.abs(true- pred) / (np.abs(true) + np.abs(pred)))
    corr = np.corrcoef(true, pred)[0, 1]
    maape=np.mean(np.true_divide(np.abs(pred-true), np.abs(true)))
    return dict(nrmse=nrmse, mae=mae, smape=smape, corr=corr, maape=maape)





def get_realibility_scores(true:np.array, q_pred:np.array, tau:np.array):
    
    #https://github.com/tony-psq/QRMGM_KDE/blob/master/QRMGM_KDE/evaluation/Evaluation.py
    assert true.ndim == 1, "pred must be 1-dimensional"
    assert q_pred.ndim == 2, "pred must be 1-dimensional"
    assert tau.ndim == 2, "pred must be 1-dimensional"
    assert tau.shape == q_pred.shape, "pred and true must have the same shape"
    assert len(true) == q_pred.shape[0], "pred and true must have the same shape"
    
    y_cdf = np.zeros((q_pred.shape[0], q_pred.shape[1] + 2))
    y_cdf[:, 1:-1] = q_pred
    y_cdf[:, 0] = 2.0 * q_pred[:, 1] - q_pred[:, 2]
    y_cdf[:, -1] = 2.0 * q_pred[:, -2] - q_pred[:, -3]
    
    
    qs = np.zeros((q_pred.shape[0], q_pred.shape[1] + 2))
    qs[:, 1:-1] = tau
    qs[:, 0] = 0.0
    qs[:, -1] = 1.0
    
    PIT = np.zeros(true.shape)
    for i in range(true.shape[0]):
        PIT[i] = np.interp(np.squeeze(true[i]), np.squeeze(y_cdf[i, :]), np.squeeze(qs[i, :]))
        
    return PIT
        
    



def get_quantile_crps_scores(true:np.array, 
                    q_pred:np.array, 
                    tau:np.array,      
                    target_range:float=None):
    """calculate prediction interval scores

    Args:
        pred (np.array): predicted values
        true (np.array): true values
        q_pred (float): prediction interval
        target_range (float): target range

    Returns:
        [pic(float), nmpi(float), nrmsq(float), ncrsp(float)]: prediction interval scores
    """

    
    assert true.ndim == 1, "pred must be 1-dimensional"
    assert q_pred.ndim == 2, "pred must be 1-dimensional"
    assert tau.ndim == 2, "pred must be 1-dimensional"
    assert tau.shape == q_pred.shape, "pred and true must have the same shape"
    assert len(true) == q_pred.shape[0], "pred and true must have the same shape"
    
    y_cdf = np.zeros((q_pred.shape[0], q_pred.shape[1] + 2))
    y_cdf[:, 1:-1] = q_pred
    y_cdf[:, 0] = 2.0 * q_pred[:, 1] - q_pred[:, 2]
    y_cdf[:, -1] = 2.0 * q_pred[:, -2] - q_pred[:, -3]
    
    
    qs = np.zeros((q_pred.shape[0], q_pred.shape[1] + 2))
    qs[:, 1:-1] = tau
    qs[:, 0] = 0.0
    qs[:, -1] = 1.0
    
    
    ind = np.zeros(y_cdf.shape)
    ind[y_cdf > true.reshape(-1, 1)] = 1.0
    CRPS = np.trapz((qs - ind) ** 2.0, y_cdf)
    CRPS = np.mean(CRPS)
    if target_range is not None:
        CRPS = CRPS/target_range
    return CRPS

  

def get_prediction_interval_scores(pred:np.array, 
                                    true:np.array, 
                                    q_pred:np.array, 
                                    target_range:float, 
                                    samples:np.array=None,
                                    tau:np.array=None):
    """calculate prediction interval scores

    Args:
        pred (np.array): predicted values
        true (np.array): true values
        q_pred (float): prediction interval
        target_range (float): target range

    Returns:
        [pic(float), nmpi(float), nrmsq(float), ncrsp(float)]: prediction interval scores
    """

    assert pred.ndim == 1, "pred must be 1-dimensional"
    assert true.ndim == 1, "pred must be 1-dimensional"
    assert q_pred.ndim == 2, "pred must be 1-dimensional"
    assert pred.shape == true.shape, "pred and true must have the same shape"

    #get prediction interval for SSM
    #target_#range = true.max() - true.min()
    residuals = pred - true
   
    
    if samples is not None:
        nrmseq=np.mean([np.sqrt(mean_squared_error(true, q))/target_range for q in samples])
        ncrps = eval_crps(torch.tensor(samples).unsqueeze(-1), torch.tensor(true).unsqueeze(-1))
        

        mu  = samples.mean(0)
        std = samples.std(0)

        sharpness = np.sqrt(np.mean(std ** 2))/target_range

        lower, upper = mu-std, mu+std
        #lower, upper = q_pred[0], q_pred[-1]
        scores = np.mean(((pred-lower)/target_range + (upper-pred)/target_range)*0.5)

    #get prediction interval for FPQ
    else:
        lower, upper = q_pred[0], q_pred[-1]
        nrmseq=np.mean([np.sqrt(mean_squared_error(true, q))/target_range for q in q_pred])
        ncrps = eval_crps(torch.tensor(q_pred).unsqueeze(-1), torch.tensor(true).unsqueeze(-1))
        
        std = q_pred.std(0)

        sharpness = np.sqrt(np.mean(std ** 2))/target_range
        l_25, u_75 =  q_pred[int(len(q_pred)*0.1)], q_pred[int(len(q_pred)*0.965)]
        

        scores = np.maximum((pred-l_25), (pred-u_75))


    # Compute nll
    nll_list = stats.norm.logpdf(residuals, scale=std)
    nll = -1 * np.sum(nll_list)/ len(true)

    #get prediction interval probability
    pic = np.intersect1d(np.where(true > lower)[0], np.where(true < upper)[0])
    pic = len(pic)/len(true)

    #get nmpi
    diffs = np.maximum(0, upper-lower)
    nmpic = diffs.mean()/target_range

    

    qncrps = get_quantile_crps_scores(true, q_pred.T, tau.T, None)

    return dict(pic=pic, nmpi=nmpic, nrmseq=nrmseq, ncrps=ncrps, sharpness=sharpness,  nll= nll, qncrps=qncrps, scores=scores)







def get_combined_CIWPscore(nmpi:float, pic:float, nrmse:float,  true_nmpic:float, alpha:float=1.0):
    """calculate combined CIPWRMSE score

    Args:
        nmpi (float): nmpi score
        pic (float): p
        nrmse (float): nrmse score
        true_nmpic (float): true nmpic score

    Returns:
        [score(float)]: combined CIPWRMSE score
    """
    #get nmpi difference
    nmpic_diff = np.abs(true_nmpic-nmpi)
    pic_diff = np.abs(alpha-pic)
    #get pic score
   
    #pic_score=(np.exp(-nrmse*pic_diff))*pic
    
    
    #get nmpi score
    error  = max((1-nrmse), 0)
    pic_score=error*np.exp(-pic_diff)/(1+pic_diff)
    nmpic_score = error*np.exp(-nmpic_diff)/(1+np.abs(nmpic_diff))
    #get nmpi score
    num = 2*nmpic_score*pic_score
    denom = (pic_score + nmpic_score)
    score=np.true_divide(num, denom)
    #if denom<=0.0:
    #    denom=1.0
    #score = np.divide(num, denom)
    return score


def get_daily_metrics(pred:np.array, true:np.array,  q_pred:np.array, target_range:float, true_nmpic:float, samples=None, tau=None, alpha=0.5):
    """calculate daily metrics

    Args:
        pred (np.array): predicted values
        true (np.array): true values
        target_range (float): target range
        q_pred (np.array): prediction interval

    Returns:
        [daily_metrics(dict)]: daily metrics
    """
    assert pred.ndim == 1, "pred must be 1-dimensional"
    assert true.ndim == 1, "pred must be 1-dimensional"
    assert q_pred.ndim == 2, "pred must be 1-dimensional"
    assert pred.shape == true.shape, "pred and true must have the same shape"

    #get pointwise metrics
    metrics = get_pointwise_metrics(pred, true, target_range)
    #get prediction interval scores
    prediction_interval_scores = get_prediction_interval_scores(pred, true, q_pred, target_range, samples, tau)
    #get combined CWE
    ciwe = get_combined_CIWPscore(prediction_interval_scores['nmpi'], 
                                                    prediction_interval_scores['pic'], 
                                                    prediction_interval_scores['nrmseq'], 
                                                    true_nmpic, alpha)

    ciwcov = get_combined_CIWPscore(prediction_interval_scores['nmpi'], 
                                                    prediction_interval_scores['pic'], 
                                                    1-metrics['corr'], 
                                                    true_nmpic, alpha)

    ciwf = get_combined_CIWPscore(prediction_interval_scores['nmpi'], 
                                                    prediction_interval_scores['pic'], 
                                                    prediction_interval_scores['ncrps']/target_range, 
                                                    true_nmpic, alpha)

    ciwfq = get_combined_CIWPscore(prediction_interval_scores['nmpi'], 
                                                    prediction_interval_scores['pic'], 
                                                    prediction_interval_scores['qncrps']/target_range, 
                                                    true_nmpic, alpha)

    metrics.update(prediction_interval_scores)
    metrics['ciwf'] = ciwf
    metrics['ciwe'] = ciwe
    metrics['ciwfq'] = ciwfq
    metrics['ciwcov'] = ciwcov 
    metrics =pd.DataFrame.from_dict(metrics, orient='index').T

    return metrics



def get_daily_pointwise_metrics(pred:np.array, true:np.array, target_range:float):
    assert pred.ndim == 1, "pred must be 1-dimensional"
    assert true.ndim == 1, "pred must be 1-dimensional"
    assert pred.shape == true.shape, "pred and true must have the same shape"

    #get pointwise metrics
    metrics = get_pointwise_metrics(pred, true, target_range)
    metrics =pd.DataFrame.from_dict(metrics, orient='index').T
    return metrics
    





