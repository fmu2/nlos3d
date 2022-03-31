import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def mse_loss(pred, target, reduction="sum"):
    return F.mse_loss(pred.flatten(), target.flatten(), reduction=reduction)

def l1_loss(pred, target, reduction="sum"):
    return F.l1_loss(pred.flatten(), target.flatten(), reduction=reduction)

def poisson_nll_loss(pred, target, reduction="sum"):
    return F.poisson_nll_loss(
        pred.flatten(), target.flatten(), log_input=False, reduction=reduction
    )

def tv_loss(x, log_space=True, mode="l1", eps=0.1, reduction="sum"):
    if log_space:
        logx = torch.log(eps + x)
        diff = logx[..., 1:] - logx[..., :-1]
    else:
        diff = x[..., 1:] - x[..., :-1]

    if mode == "l1":
        loss = torch.sum(torch.abs(diff), -1)
    elif mode == "l2":
        loss = torch.sqrt(eps + torch.sum(diff ** 2, -1))
    else:
        raise NotImplementedError("invalid TV mode: {:s}".format(mode))
    if reduction == "mean":
        return torch.mean(loss)
    else:
        return torch.sum(loss)

def beta_loss(x, log_space=True, eps=0.1, reduction="sum"):
    if log_space:
        loss = torch.log(eps + x) + torch.log(eps + 1 - x)
    else:
        loss = x + torch.log(eps - torch.expm1(x))
    if reduction == "mean":
        return torch.mean(loss)
    else:
        return torch.sum(loss)