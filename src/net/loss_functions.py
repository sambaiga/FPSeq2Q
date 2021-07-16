import torch
import numpy as np
#https://github.com/ku2482/fqf-iqn-qrdqn.pytorch

def calculate_huber_loss(td_errors, kappa=1.0):
    return torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa))


def calculate_quantile_huber_loss(td_errors, taus, weights=None, kappa=1.0):
    assert not taus.requires_grad
    batch_size, N, N_dash = td_errors.shape
    # Calculate huber loss element-wisely.
    element_wise_huber_loss = calculate_huber_loss(td_errors, kappa)
    assert element_wise_huber_loss.shape == (
        batch_size, N, N_dash)

    # Calculate quantile huber loss element-wisely.
    element_wise_quantile_huber_loss = torch.abs(
        taus - (td_errors.detach() < 0).float()
        ) * element_wise_huber_loss / kappa
    assert element_wise_quantile_huber_loss.shape == (
        batch_size, N, N_dash)

    # Quantile huber loss.
    batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(
        dim=1).mean(dim=1, keepdim=True)
    assert batch_quantile_huber_loss.shape == (batch_size, 1)

    if weights is not None:
        quantile_huber_loss = (batch_quantile_huber_loss * weights).mean()
    else:
        quantile_huber_loss = batch_quantile_huber_loss.mean()

    return quantile_huber_loss




def cwi_score(y, quantile_hats, eps=1e-6):
    with torch.no_grad():
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





def N_quantile_proposal_loss(quantile, quantile_hats, taus):
    
    assert not taus.requires_grad
    assert not quantile_hats.requires_grad
    assert not quantile.requires_grad

    value_1 = quantile - quantile_hats[:, :-1]
    signs_1 = quantile > torch.cat([quantile_hats[:, :1,:], quantile[:, :-1,:]], dim=1)
    value_2 = quantile - quantile_hats[:, 1:]
    signs_2 = quantile < torch.cat([quantile[:, 1:], quantile_hats[:, -1:]], dim=1)
    gradient_tau = (torch.where(signs_1, value_1, -value_1) + torch.where(signs_2, value_2, -value_2)).view(*value_1.size())
    tau_loss = torch.mul(gradient_tau.detach(), taus[:, 1: -1, :]).sum(1).mean()
    return tau_loss





