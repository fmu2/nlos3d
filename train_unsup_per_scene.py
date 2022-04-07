import os
import argparse

import yaml

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from libs.config import load_config
from libs.data import make_measurement, make_images
from libs.worker import UnsupRendererWorker
from libs.optimizer import *
from libs.utils import *


def main(args):

    # set up checkpoint folder
    os.makedirs('ckpt', exist_ok=True)
    ckpt_path = os.path.join('ckpt', args.name)
    ensure_path(ckpt_path)

    # load config
    try:
        config_path = os.path.join(ckpt_path, 'config.yaml')
        check_file(config_path)
        config = load_config(config_path, mode='train_unsup_per_scene')
        print('config loaded from checkpoint folder')
        config['_resume'] = True
    except:
        check_file(args.config)
        config = load_config(args.config, mode='train_unsup_per_scene')
        print('config loaded from command line')

    # configure GPUs
    set_gpu(args.gpu)
    n_gpus = torch.cuda.device_count()

    set_log_path(ckpt_path)
    writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))
    rng = fix_random_seed(config.get('seed', 2022))

    ###########################################################################
    """ worker """

    itr0 = 0
    n_itrs = config['opt']['n_itrs']
    if config.get('_resume'):
        ckpt_name = os.path.join(ckpt_path, 'last.pth')
        try:
            check_file(ckpt_name)
            ckpt = torch.load(ckpt_name)
            
            worker = UnsupRendererWorker(
                wall_cfg=ckpt['config']['wall'],
                cam_cfg=ckpt['config']['camera'],
                model_cfg=ckpt['config']['model'],
            )
            worker.load(ckpt)
            worker.cuda(n_gpus)
            
            optimizer = load_optimizer(worker, ckpt)
            scheduler = load_scheduler(optimizer, ckpt)

            itr0, config = ckpt['itr'], ckpt['config']
        except:
            config.pop('_resume')
            itr0 = 0

            worker = UnsupRendererWorker(
                wall_cfg=config['wall'],
                cam_cfg=config['camera'],
                model_cfg=config['model'],
            )
            worker.cuda(n_gpus)

            optimizer = make_optimizer(worker, config['opt'])
            scheduler = make_scheduler(optimizer, config['opt'])
    else:
        worker = UnsupRendererWorker(
            wall_cfg=config['wall'],
            cam_cfg=config['camera'],
            model_cfg=config['model'],
        )
        worker.cuda(n_gpus)

        optimizer = make_optimizer(worker, config['opt'])
        scheduler = make_scheduler(optimizer, config['opt'])

        yaml.dump(config, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))
    
    print('worker initialized, train from itr {:d}'.format(itr0 + 1))
    print('number of model parameters: {:s}'.format(count_params(worker)))

    ############################################################################
    """ dataset """

    meas = make_measurement(config['meas'])         # (1/3, t, h, w)
    gt = make_images(config['target'])              # (v, 1/3, h, w)

    ############################################################################
    """ Training / Validation """

    loss_list = ['poisson', 'beta', 'tv']
    train_losses = {k: AverageMeter() for k in loss_list}

    metric_list = ['rmse', 'psnr', 'ssim']

    timer = Timer()
    
    for itr in range(itr0 + 1, n_itrs + 1):
        loss_dict, _ = worker.train(
            meas=meas,
            cfg=config['train'],
        )

        loss = config['opt']['poisson'] * loss_dict['poisson'] \
             + config['opt']['beta'] * loss_dict['beta'] \
             + config['opt']['tv'] * loss_dict['tv']

        for k in loss_dict.keys():
            train_losses[k].update(loss_dict[k].item())
            writer.add_scalars(k, {'train': train_losses[k].item()}, itr)

        optimizer.zero_grad()
        loss.backward()

        if config['opt']['clip_grad_norm'] > 0:
            nn.utils.clip_grad_norm_(
                worker.parameters(), config['opt']['clip_grad_norm']
            )
        
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], itr)
        
        optimizer.step()
        scheduler.step()

        if itr % args.print_freq == 0 or itr == 1:
            torch.cuda.synchronize()
            t_elapsed = time_str(timer.end())

            log_str = '[{:03d}/{:03d}] '.format(
                itr // args.print_freq, n_itrs // args.print_freq
            )
            for k in loss_list:
                log_str += '{:s} {:.3f} | '.format(k, train_losses[k].item())
            log_str += t_elapsed
            log(log_str, 'log.txt')

            writer.flush()
            for k in loss_list:
                train_losses[k].reset()

            ckpt = worker.save()
            ckpt['itr'] = itr
            ckpt['config'] = config
            ckpt['optimizer'] = optimizer.state_dict()
            ckpt['scheduler'] = scheduler.state_dict()
            torch.save(ckpt, os.path.join(ckpt_path, 'last.pth'))
            timer.start()

        if itr % args.val_freq == 0:
            loss, output_dict, metric_dict = worker.eval(
                target=gt,
                cfg=config['eval'],
            )

            writer.add_scalars('mse', {'val': loss.item()}, itr)
            for k in metric_dict.keys():
                writer.add_scalars(k, {'val': metric_dict[k].item()}, itr)

            pred = output_dict['pred']
            target = output_dict['target']
            pred = torch.clamp(pred, 0, 1)
            target = torch.clamp(target, 0, 1)

            img = torch.stack([pred, target], dim=1).flatten(0, 1)
            if img.dim() == 3:
                img = img.unsqueeze(1)
            writer.add_images(
                tag='val/{:03d}'.format(itr),
                img_tensor=img,
                global_step=itr,
            )

            t_elapsed = time_str(timer.end())
            log_str = '[{:03d}/{:03d} val] '.format(
                itr // args.print_freq, n_itrs // args.print_freq
            )
            log_str += 'loss: {:.3f} | '.format(loss.item())
            for k in metric_list:
                log_str += '{:s} {:.2f} | '.format(k, metric_dict[k].item())
            log_str += t_elapsed
            log(log_str, 'log.txt')

            writer.flush()

            ckpt = worker.save()
            ckpt['itr'] = itr
            ckpt['config'] = config
            ckpt['optimizer'] = optimizer.state_dict()
            ckpt['scheduler'] = scheduler.state_dict()
            torch.save(ckpt, os.path.join(ckpt_path, '{:d}.pth'.format(n_itrs)))

            timer.start()

################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config file')
    parser.add_argument('-n', '--name', help='job name')
    parser.add_argument('-g', '--gpu', type=str, default='0', 
                        help='GPU device IDs')
    parser.add_argument('-pf', '--print_freq', type=int, default=1, 
                        help='print frequency (x100 itrs)')
    parser.add_argument('-vf', '--val_freq', type=int, default=1,
                        help='validation frequency (x100 itrs)')
    args = parser.parse_args()

    args.print_freq *= 100
    args.val_freq *= 100

    main(args)