import argparse
import random
import sys
import os
from datetime import datetime
import numpy as np
import logging
import torch
import torch.optim as optim

from models.models import VRIC_IAT
from dataset import get_dataloader
from utils import init, Logger, load_checkpoint, save_checkpoint, AverageMeter
from losses.losses import Metrics, PixelwiseRateDistortionLoss

logger_handle = logging.getLogger("HiFi_Variable-Rate_Image_Compression_via_IAT")
def parse_args(argv):
    parser = argparse.ArgumentParser(description='HiFi_Variable-Rate_Image_Compression_via_IAT')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--name', help='result dir name', default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), type=str)
    parser.add_argument('--resume', help='snapshot path', type=str)
    parser.add_argument('--seed', help='seed number', default=None, type=int)
    args = parser.parse_args(argv)

    if not args.config:
        if args.resume:
            assert args.resume.startswith('./')
            dir_path = '/'.join(args.resume.split('/')[:-2])
            args.config = os.path.join(dir_path, 'config.yaml')
        else:
            args.config = './configs/config.yaml'

    return args


# V in the paper
def quality2lambda(qlevel):
    return 1e-3 * torch.exp(4.382 * qlevel)


def test(logger, test_dataloaders, model, criterion, metric):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()

    with torch.no_grad():
        for i, test_dataloader in enumerate(test_dataloaders):
            logger.init()
            for x, qlevel in test_dataloader:
                x = x.to(device)
                qlevel = qlevel.to(device)
                lmbdalevel = quality2lambda(qlevel)
                out_net = model(x, qlevel)
                out_net['x_hat'].clamp_(0, 1)

                out_criterion = criterion(out_net, x, lmbdalevel)
                bpp, psnr, ms_ssim = metric(out_net, x)

                logger.update_test(bpp, psnr, ms_ssim, out_criterion, model.aux_loss())
            level = i-1
            logger.print_test(level)
            logger.write_test(level)
            if level != -1:
                loss.update(logger.loss.avg)
                bpp_loss.update(logger.bpp_loss.avg)
                mse_loss.update(logger.mse_loss.avg)
        logger_handle.info(f'[ Test ] Total mean: {loss.avg:.4f}')
    logger.init()
    model.train()

    return loss.avg, bpp_loss.avg, mse_loss.avg


def train(args, config, base_dir, snapshot_dir, output_dir, log_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = PixelwiseRateDistortionLoss()
    metric = Metrics()
    train_dataloader, test_dataloaders = get_dataloader(config)
    logger = Logger(config, base_dir, snapshot_dir, output_dir, log_dir, logger_handle=logger_handle)

    model = VRIC_IAT(N=config['N'], M=config['M'])
    model = model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)/1024./1024.
    logger_handle.info(f'number of params: {n_parameters:.6f}M flops')

    # from thop import profile
    # input_x = torch.randn(1, 3, 512, 512).to(device)
    # input_x1 = torch.randn(1, 1, 512, 512).to(device)
    # flops, params = profile(model, inputs=(input_x, input_x1))
    # flops_m = flops/1024./1024./1024.
    # params_m = params/1024./1024.
    # logger_handle.info(f"flops:{flops_m}G, params:{params_m}M")


    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    aux_optimizer = optim.Adam(model.aux_parameters(), lr=config['lr_aux'])

    if args.resume:
        itr, model = load_checkpoint(args.resume, model, optimizer, aux_optimizer)
        logger.load_itr(itr)

    if config['set_lr']:
        lr_prior = optimizer.param_groups[0]['lr']
        for g in optimizer.param_groups:
            g['lr'] = float(config['set_lr'])
        logger_handle.info(f'[set lr] {lr_prior} -> {optimizer.param_groups[0]["lr"]}')
    model.train()
    loss_best = 1e10
    while logger.itr < config['max_itr']:
        for x, qlevel in train_dataloader:
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            x = x.to(device)
            qlevel = qlevel.to(device)
            lmbdalevel = quality2lambda(qlevel)
            out_net = model(x, qlevel)
            out_criterion = criterion(out_net, x, lmbdalevel)
            out_criterion['loss'].backward()
            aux_loss = model.aux_loss()
            aux_loss.backward()

            # for stability
            if out_criterion['loss'].isnan().any() or out_criterion['loss'].isinf().any() or out_criterion['loss'] > 10000:
                logger_handle.info(f"out_criterion['loss'].isnan().any():{out_criterion['loss'].isnan().any()} \
                                     out_criterion['loss'].isinf().any():{out_criterion['loss'].isinf().any()} \
                                     out_criterion['loss'] > 10000:{out_criterion['loss'] > 10000}")
                continue

            if config['clip_max_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_max_norm'])
            optimizer.step()
            aux_optimizer.step()  # update quantiles of entropy bottleneck modules

            # logging
            logger.update(out_criterion, aux_loss)
            if logger.itr % config['log_itr'] == 0:
                logger.print()
                logger.write()
                logger.init()
            # test and save model snapshot
            if logger.itr % config['test_itr'] == 0 or logger.itr % config['snapshot_save_itr'] == 0:
                # model.module.update()
                model.update()
                loss, bpp_loss, mse_loss = test(logger, test_dataloaders, model, criterion, metric)
                if loss < loss_best:
                    logger_handle.info('Best!')
                    save_checkpoint(os.path.join(snapshot_dir, 'best.pt'), logger.itr, model, optimizer, aux_optimizer)
                    loss_best = loss
                if logger.itr % config['snapshot_save_itr'] == 0:
                    save_checkpoint(os.path.join(snapshot_dir, f'{logger.itr:07}_{bpp_loss:.4f}_{mse_loss:.8f}.pt'),
                                    logger.itr, model, optimizer, aux_optimizer)

            if logger.itr % config['lr_shedule_step'] == 0 or logger.itr > config['lr_shedule_step']:
                if logger.itr % config['lr_shedule_step'] == 0:
                    lr_prior = optimizer.param_groups[0]['lr']
                    for g in optimizer.param_groups:
                        g['lr'] *= config['lr_shedule_scale']
                    logger_handle.info(f'[lr scheduling] {lr_prior} -> {optimizer.param_groups[0]["lr"]}')
                elif (logger.itr - config['lr_shedule_step']) % 300000 == 0:
                    lr_prior = optimizer.param_groups[0]['lr']
                    for g in optimizer.param_groups:
                        g['lr'] *= config['lr_shedule_scale']
                    logger_handle.info(f'[lr scheduling] {lr_prior} -> {optimizer.param_groups[0]["lr"]}')

def main(argv):
    args = parse_args(argv)
    config, base_dir, snapshot_dir, output_dir, log_dir = init(args)
    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU

    formatter = logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger_handle.addHandler(stdhandler)
    if log_dir != '':
        filehandler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger_handle.addHandler(filehandler)
    logger_handle.setLevel(logging.INFO)

    logger_handle.info(f'[PID:{os.getpid()}]')
    logger_handle.info(f'[config:{args.config}]')


    msg = f'======================= {args.name} ======================='
    logger_handle.info(msg)
    for k, v in config.items():
        if k in {'lr', 'set_lr', 'p'}:
            logger_handle.info(f' *{k}: {v}')
        else:
            logger_handle.info(f' *{k}: {v}')
    logger_handle.info('=' * len(msg))
    train(args, config, base_dir, snapshot_dir, output_dir, log_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
