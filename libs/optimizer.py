import torch.optim as optim
import torch.optim.lr_scheduler as sched


def make_optimizer(model, cfg):
    if cfg['optim_type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg['lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'],
        )
    elif cfg['optim_type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg['lr'],
            betas=(cfg['beta1'], cfg['beta2']),
            weight_decay=cfg['weight_decay'],
        )
    elif cfg['optim_type'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg['lr'],
            betas=(cfg['beta1'], cfg['beta2']),
            weight_decay=cfg['weight_decay'],
        )
    else:
        raise NotImplementedError(
            'invalid optimizer: {:s}'.format(cfg['optim_type'])
        )

    return optimizer


def make_scheduler(optimizer, cfg):
    if cfg['sched_type'] == 'cosine':
        scheduler = sched.CosineAnnealingLR(
            optimizer, 
            T_max=cfg['n_itrs'],
        )
    elif cfg['sched_type'] == 'step':
        scheduler = sched.MultiStepLR(
            optimizer, 
            milestones=cfg['milestones'], 
            gamma=cfg['gamma'],
        )
    elif cfg['sched_type'] == 'exp':
        scheduler = sched.LambdaLR(
            optimizer, 
            lr_lambda=lambda i: cfg['gamma'] ** (i / cfg['decay_steps']),
        )
        raise NotImplementedError(
            'invalid scheduler: {:s}'.format(cfg['sched_type'])
        )

    return scheduler


def load_optimizer(model, ckpt):
    cfg = ckpt['config']['opt']
    optimizer = make_optimizer(model, cfg)
    optimizer.load_state_dict(ckpt['optimizer'])
    return optimizer


def load_scheduler(optimizer, ckpt):
    cfg = ckpt['config']['opt']
    scheduler = make_scheduler(optimizer, cfg)
    scheduler.load_state_dict(ckpt['scheduler'])
    return scheduler