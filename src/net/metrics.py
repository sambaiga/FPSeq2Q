import math
import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    mean_absolute_percentage_error
)
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
from tqdm import tqdm
from scipy import stats
import pandas as pd


def sharpness(y_std):
    """
    Return sharpness (a single measure of the overall confidence).
    """

    # Compute sharpness
    sharp_metric = np.sqrt(np.mean(y_std ** 2))

    return sharp_metric


def root_mean_squared_calibration_error(
    y_pred, y_std, y_true, num_bins=100, vectorized=False, recal_model=None
):
    """Return root mean squared calibration error."""

    # Get lists of expected and observed proportions for a range of quantiles
    if vectorized:
        (exp_proportions, obs_proportions) = get_proportion_lists_vectorized(
            y_pred, y_std, y_true, num_bins, recal_model
        )
    else:
        (exp_proportions, obs_proportions) = get_proportion_lists(
            y_pred, y_std, y_true, num_bins, recal_model
        )

    squared_diff_proportions = np.square(exp_proportions - obs_proportions)
    rmsce = np.sqrt(np.mean(squared_diff_proportions))

    return rmsce


def mean_absolute_calibration_error(
    y_pred, y_std, y_true, num_bins=100, vectorized=False, recal_model=None
):
    """ Return mean absolute calibration error; identical to ECE. """

    # Get lists of expected and observed proportions for a range of quantiles
    if vectorized:
        (exp_proportions, obs_proportions) = get_proportion_lists_vectorized(
            y_pred, y_std, y_true, num_bins, recal_model
        )
    else:
        (exp_proportions, obs_proportions) = get_proportion_lists(
            y_pred, y_std, y_true, num_bins, recal_model
        )

    abs_diff_proportions = np.abs(exp_proportions - obs_proportions)
    mace = np.mean(abs_diff_proportions)

    return mace


#https://github.com/uncertainty-toolbox/uncertainty-toolbox/blob/master/uncertainty_toolbox/metrics_accuracy.py
def prediction_error_metrics(y_pred, y_true):
    """
    Return prediction error metrics as a dict with keys:
    - Mean average error ('mae')
    - Root mean squared error ('rmse')
    - Median absolute error ('mdae')
    - Mean absolute relative percent difference ('marpd')
    - r^2 ('r2')
    - Pearson's correlation coefficient ('corr')
    """

    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mdae = median_absolute_error(y_true, y_pred)
    residuals = y_true - y_pred
    marpd = np.abs(2 * residuals / (np.abs(y_pred) + np.abs(y_true))).mean() * 100
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    prediction_metrics = {
        "mae": mae,
        "rmse": rmse,
        "mdae": mdae,
        "marpd": marpd,
        "r2": r2,
        "corr": corr,
    }

    return prediction_metrics


def miscalibration_area(
    y_pred, y_std, y_true, num_bins=100, vectorized=False, recal_model=None
):
    """
    Return miscalibration area.
    This is identical to mean absolute calibration error and ECE, however
    the integration here is taken by tracing the area between curves.
    In the limit of num_bins, miscalibration area and
    mean absolute calibration error will converge to the same value.
    """

    # Get lists of expected and observed proportions for a range of quantiles
    if vectorized:
        (exp_proportions, obs_proportions) = get_proportion_lists_vectorized(
            y_pred, y_std, y_true, num_bins, recal_model
        )
    else:
        (exp_proportions, obs_proportions) = get_proportion_lists(
            y_pred, y_std, y_true, num_bins, recal_model
        )

    # Compute approximation to area between curves
    polygon_points = []
    for point in zip(exp_proportions, obs_proportions):
        polygon_points.append(point)
    for point in zip(reversed(exp_proportions), reversed(exp_proportions)):
        polygon_points.append(point)
    polygon_points.append((exp_proportions[0], obs_proportions[0]))
    polygon = Polygon(polygon_points)
    x, y = polygon.exterior.xy
    ls = LineString(np.c_[x, y])
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    mls = unary_union(lr)
    polygon_area_list = [poly.area for poly in polygonize(mls)]
    miscalibration_area = np.asarray(polygon_area_list).sum()

    return miscalibration_area


def get_proportion_lists_vectorized(
    y_pred, y_std, y_true, num_bins=100, recal_model=None
):
    """
    Return lists of expected and observed proportions of points falling into
    intervals corresponding to a range of quantiles.
    """

    # Compute proportions
    exp_proportions = np.linspace(0, 1, num_bins)
    # If we are recalibrating, input proportions are recalibrated proportions
    if recal_model is not None:
        in_exp_proportions = recal_model.predict(exp_proportions)
    else:
        in_exp_proportions = exp_proportions

    norm = stats.norm(loc=0, scale=1)
    gaussian_lower_bound = norm.ppf(0.5 - in_exp_proportions / 2.0)
    gaussian_upper_bound = norm.ppf(0.5 + in_exp_proportions / 2.0)
    residuals = y_pred - y_true
    normalized_residuals = (residuals.flatten() / y_std.flatten()).reshape(-1, 1)
    above_lower = normalized_residuals >= gaussian_lower_bound
    below_upper = normalized_residuals <= gaussian_upper_bound

    within_quantile = above_lower * below_upper
    obs_proportions = np.sum(within_quantile, axis=0).flatten() / len(residuals)

    return exp_proportions, obs_proportions


def get_proportion_lists(y_pred, y_std, y_true, num_bins=100, recal_model=None):
    """
    Return lists of expected and observed proportions of points falling into
    intervals corresponding to a range of quantiles.
    """

    # Compute proportions
    exp_proportions = np.linspace(0, 1, num_bins)
    # If we are recalibrating, input proportions are recalibrated proportions
    if recal_model is not None:
        in_exp_proportions = recal_model.predict(exp_proportions)
    else:
        in_exp_proportions = exp_proportions

    obs_proportions = [
        get_proportion_in_interval(y_pred, y_std, y_true, quantile)
        for quantile in in_exp_proportions
    ]

    return exp_proportions, obs_proportions


def get_proportion_in_interval(y_pred, y_std, y_true, quantile):
    """
    For a specified quantile, return the proportion of points falling into
    an interval corresponding to that quantile.
    """

    # Computer lower and upper bound for quantile
    norm = stats.norm(loc=0, scale=1)
    lower_bound = norm.ppf(0.5 - quantile / 2)
    upper_bound = norm.ppf(0.5 + quantile / 2)

    # Compute proportion of normalized residuals within lower to upper bound
    residuals = y_pred - y_true
    normalized_residuals = residuals.reshape(-1) / y_std.reshape(-1)
    num_within_quantile = 0
    for resid in normalized_residuals:
        if lower_bound <= resid <= upper_bound:
            num_within_quantile += 1.0
    proportion = num_within_quantile / len(residuals)

    return proportion






def combined_CIPWRMSEscore(nmpi, pic, nrmse=0.063,  true_nmpic=0.392):
    nmpic_diff = np.abs(true_nmpic-nmpi)
    pic_diff   = 1-pic
    pic_score=(np.exp(-nrmse*pic_diff))*pic
    nmpic_score = np.exp(-nrmse*nmpic_diff)/(1+np.abs(nmpic_diff))
    score = 2*nmpic_score*pic_score/(pic_score + nmpic_score)
    return score*pic


def get_prediction_interval_scores(mu, true, q_pred, R=None):
    
    lower, upper = q_pred[0], q_pred[-1]
    pic = np.intersect1d(np.where(true > lower)[0], np.where(true < upper)[0])
    pic = len(pic)/len(true)
    
    diffs = np.maximum(0, upper-lower)
    
    if R is None:
        R = true.max() - true.min()
    nmpic = diffs.sum()/(R*len(true))

    return pic, nmpic



def cwi_score(y, quantile_hats, eps=1e-6):
    lower, upper = quantile_hats[:,0, :], quantile_hats[:,-1,:]
    pic = np.intersect1d(np.where(y.data.cpu().numpy().flatten() > lower.data.cpu().numpy().flatten())[0], np.where(y.data.cpu().numpy().flatten() < upper.data.cpu().numpy().flatten())[0])
    pic = torch.tensor(len(pic)/len(y.flatten())).to(y.device)
    mpic =  (upper-lower).abs().mean()
    
    y_q    = y.unsqueeze(1).expand_as(quantile_hats)
    nrmse_q = torch.sqrt((y_q - quantile_hats)**2).sum(axis=1).mean()
    nrmse = torch.nn.functional.mse_loss(quantile_hats, y_q, reduction='none').sum(1).mean()

    true_mpic = 2*y.std()
    mpic_diff = (true_mpic-mpic).abs()
    pic_diff   = 1-pic
            
    pic_score=(torch.exp(-nrmse*pic_diff))*pic
    mpic_score = torch.exp(-nrmse*mpic_diff)/(1+ mpic_diff)
    
    score = torch.div(2*mpic_score*pic_score, (pic_score + mpic_score)+eps)
    #score = torch.nan_to_num(score)
    
    return 1-score



def get_metrics_dataframe(true, mu, q_pred=None, tau_hat=None, a_step=48, R=None, true_nmpic=None):

    if R is None:
        R = true.max() - true.min() 

    
    nrmse = []
    pic = []
    nmpi = []
    mape = []
    ciwrmse = []
    
    true_nmpic = []
    UCE = []
    epistemitic = []
    aelotic = []
    uncert = []
    error = []
    
    maap = []
    mae = []
    mdae = []
    
    r2 = []
    corr = []
    marpd = []
    for day in range(0, true.shape[0], a_step):
        temp_nrmse=np.sqrt(mean_squared_error(true[day], mu[day]))/R
        nrmse.append(temp_nrmse)
        NRMSE=np.sqrt(mean_squared_error(true[day], mu[day]))/R  
        temp_mape = mean_absolute_percentage_error(true[day], mu[day])
        if q_pred is not None:
            temp_pic, temp_nmpic = get_prediction_interval_scores(mu[day], true[day], q_pred[day], R=R)
            pic.append(temp_pic)
            nmpi.append(temp_nmpic)
            nrmse_q = np.sqrt((true[day]-q_pred[day])**2).sum(axis=1).mean()/R
            temp_ciw = combined_CIPWRMSEscore(temp_nmpic, temp_pic, nrmse=nrmse_q,  true_nmpic=2*true[day].std()/R)
            ciwrmse.append(temp_ciw)

            #unc, ael, ep = get_uncertainity(torch.tensor(mu[day]), torch.tensor(q_pred[day]), torch.tensor(tau_hat[day]), dim=0)
            #err = torch.nn.functional.mse_loss(torch.tensor(q_pred[day]), torch.tensor(true[day]).unsqueeze(0).expand_as(torch.tensor(q_pred[day])), reduction='none').mean(dim=0)
            #uce, err_in_bin, avg_uncert_in_bin, prop_in_bin=uceloss(err, unc, n_bins=a_step, outlier=0.0, range=None)
            #uncert.append(unc.mean().item())
            #aelotic.append(ael.mean().item())
            #epistemitic.append(ep.mean().item())
            #UCE.append(uce.item())
            #error.append(err.mean().item())
            true_nmpic.append(2*true[day].std()/R)

        
        mape.append(min(temp_mape, 1))
        

        pred=prediction_error_metrics(mu[day], true[day])
        maap.append(min(np.arctan(np.abs((true[day]-mu[day])/true[day])).mean(), 1))
        mae.append(pred['mae'])
        mdae.append(pred['mdae'])

        corr.append(pred['corr'])
        r2.append(pred['r2'])
        marpd.append(pred['marpd'])
        
    if q_pred is None:
        metrics=dict(nrmse=nrmse,   mape=mape,  corr=corr, r2=r2, marpd=marpd, mdae=mdae, 
                     mae=mae, maap=maap)
    else:
        
        metrics=dict(nrmse=nrmse,  pic=pic, 
                     nmpi=nmpi, mape=mape, ciwrmse = ciwrmse, true_nmpic=true_nmpic, corr=corr, r2=r2, marpd=marpd, mdae=mdae, 
                     mae=mae, maap=maap)
    metrics=pd.DataFrame.from_dict(metrics)
    return metrics