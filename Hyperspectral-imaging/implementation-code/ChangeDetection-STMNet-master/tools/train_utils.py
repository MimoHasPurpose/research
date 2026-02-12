# -*- coding: utf-8 -*-
import time
import math
import os

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import logging

from model.mask_utils import *


def adjust_lr_sub(lr_init, lr_gamma, optimizer, epoch, step_index):
    if epoch < 1:
        lr = 0.0001 * lr_init
    elif epoch <= step_index[0]:
        lr = lr_init
    elif epoch <= step_index[1]:
        lr = lr_init * lr_gamma
    elif epoch <= step_index[2]:
        lr = lr_init * lr_gamma ** 2
    elif epoch > step_index[2]:
        lr = lr_init * lr_gamma ** 3

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'

    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix前缀 'module.' '''
    print('remove prefix \'{}\''.format(prefix))

    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))

    if load_to_cpu == torch.device('cpu'):
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)['model']
    else:
        device = torch.cuda.current_device()  # gpu
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))['model']

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)

    return model


def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + 'log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def train_pre_batch(tru_data, model, optimizer, device, cfg, logger):
    torch.autograd.set_detect_anomaly(True)
    save_folder = cfg['save_folder']
    save_name = cfg['save_name']
    epoch_size = cfg['epoch']
    batch_size_u = cfg['batch_size_u']
    lr = cfg['lr']
    lr_step = cfg['lr_step']

    start_epoch = 0
    train_loss_save = []
    batch_num = math.ceil(len(tru_data) / batch_size_u)  # math.ceil 向上取整
    logger.info(
        'Samples: label{}/{} epoch_size:{}, batch_num: {}, lr:{}, lr_step:{}'.format(len(tru_data), batch_size_u,
                                                                                     epoch_size, batch_num, lr,
                                                                                     lr_step))
    for epoch in range(start_epoch + 1, epoch_size):
        lr, epoch_loss, epoch_time = train_one_pre_epoch(tru_data, model, optimizer, device, epoch, cfg,
                                                         train_loss_save)
        logger.info('Epoch: {}/{} || lr: {} || total_loss: {:.4f} || Epoch time: {:.4f}s'.format(epoch, epoch_size,
                                                                                                 round(lr, 6),
                                                                                                 epoch_loss / batch_num,
                                                                                                 epoch_time))

    logger.info('save final model Epoch: {}/{}'.format(epoch, epoch_size))
    save_model = dict(
        model=model.state_dict(),
        epoch=epoch_size
    )
    torch.save(save_model, os.path.join(save_folder, save_name + '.pth'))


def train_one_pre_epoch(tru_data, model, optimizer, device, epoch, cfg, train_loss_save):
    model.train()
    lr_init = cfg['lr']
    lr_gamma = cfg['lr_gamma']
    lr_step = cfg['lr_step']
    lr_adjust = cfg['lr_adjust']
    batch_size_u = cfg['batch_size_u']
    num_workers = cfg['workers_num']

    epoch_time0 = time.time()
    epoch_loss = 0

    batch_datau = DataLoader(tru_data, batch_size_u, shuffle=True, num_workers=num_workers, pin_memory=True,
                             drop_last=True)
    batch_num = math.ceil(len(tru_data) / batch_size_u)  # math.ceil 向上取整

    if lr_adjust:
        lr = adjust_lr_sub(lr_init, lr_gamma, optimizer, epoch, lr_step)
    else:
        lr = lr_init

    for batch_idx, data in enumerate(batch_datau):
        x1, x2, gt_patches, x_noisy, gt, indices = data
        x1, x2, gt_patches, x_noisy, gt = x1.to(device), x2.to(device), gt_patches.to(device), x_noisy.to(
            device), gt.to(device)
        with autocast():
            x_dec, x_rec, total_loss = model(x1, x_noisy)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

    epoch_time = time.time() - epoch_time0
    train_loss_save.append(epoch_loss / batch_num)

    return lr, epoch_loss, epoch_time
