import torch
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



def smooth_pinball_loss(error, q, tau, alpha = 0.5, kappa = 1e3, margin = 1e-2):
    #https://github.com/hatalis/smooth-pinball-neural-network/blob/master/pinball_loss.py
    #Hatalis, Kostas, et al. "A Novel Smoothed Loss and Penalty Function 
    #for Noncrossing Composite Quantile Estimation via Deep Neural Networks." arXiv preprint (2019).
    tau_error = torch.mul(error, tau.unsqueeze(2).expand_as(error))
    q_loss = (tau_error + alpha * torch.nn.functional.softplus(-error / alpha)).sum(1).mean()
    # calculate smooth cross-over penalty
    diff = q[:, 1:, :] - q[:,:-1,:]
    penalty = kappa * torch.square(torch.nn.functional.relu(margin - diff)).mean()
    loss = penalty+q_loss
    return loss


def quantile_multitaget_proposal_loss(quantile, quantile_hats, taus):
    assert not taus.requires_grad
    assert not quantile_hats.requires_grad
    assert not quantile.requires_grad


    value_1 = quantile - quantile_hats[:, :-1, :]
    signs_1 = quantile > torch.cat([quantile_hats[:, :1,:], quantile[:, :-1,:]], dim=1)
    value_2 = quantile - quantile_hats[:, 1:]
    signs_2 = quantile < torch.cat([quantile[:, 1:], quantile_hats[:, -1:]], dim=1)
    gradient_tau = (torch.where(signs_1, value_1, -value_1) + torch.where(signs_2, value_2, -value_2)).view(*value_1.size())
    
    tau_loss = torch.mul(gradient_tau.detach(), taus[:, :, 1: -1].permute(0,2,1)).sum(1).mean()
    return tau_loss


def quantile_proposal_loss(quantile, quantile_hats, taus):
    assert not taus.requires_grad
    assert not quantile_hats.requires_grad
    assert not quantile.requires_grad


    value_1 = quantile - quantile_hats[:, :-1]
    signs_1 = quantile > torch.cat([quantile_hats[:, :1,:], quantile[:, :-1,:]], dim=1)
    value_2 = quantile - quantile_hats[:, 1:]
    signs_2 = quantile < torch.cat([quantile[:, 1:], quantile_hats[:, -1:]], dim=1)
    gradient_tau = (torch.where(signs_1, value_1, -value_1) + torch.where(signs_2, value_2, -value_2)).view(*value_1.size())
    tau_loss = torch.mul(gradient_tau.detach(), taus[:, 1: -1].unsqueeze(2).expand_as(gradient_tau)).sum(1).mean()
    return tau_loss


def calibration_loss(y, quantile_hats, confidence = 0.9):
    
    y=y.unsqueeze(1).expand_as(quantile_hats)
    idx_under = y <= quantile_hats
    idx_over = ~idx_under
    coverage = torch.mean(idx_under.float(), dim=0)
    error = y - quantile_hats
    mean_diff_under = torch.mean(-1 * error * idx_under, dim=0)
    mean_diff_over = torch.mean(error * idx_over, dim=0)
    cov_under = coverage < confidence
    cov_over = ~cov_under
    loss = (cov_under * mean_diff_over) + (cov_over * mean_diff_under)
    return loss.mean()
