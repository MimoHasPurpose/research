# -*- coding: utf-8 -*-
import data.data_preprocess as data_preprocess
from data.get_dataset import get_dataset as getdata
from model.mask_utils import *


def get_tratest_set_batch(cfg, logger):
    current_dataset = cfg['current_dataset']
    train_set_num = cfg['train_set_num']
    patch_size = cfg['patch_size']

    logger.info(
        'current_dataset:{}|| train_set_num:{},patch_size:{}'.format(current_dataset, train_set_num, patch_size))

    # loadmat
    img1, img2, gt = getdata(current_dataset)
    img1 = torch.from_numpy(img1)
    img2 = torch.from_numpy(img2)
    gt = torch.from_numpy(gt)
    img1 = img1.permute(2, 0, 1)
    img2 = img2.permute(2, 0, 1)

    img1 = data_preprocess.std_norm(img1)
    img2 = data_preprocess.std_norm(img2)

    # label transform
    if current_dataset == 'Farmland':
        img_gt = gt  # Farmland(450, 140, 155), gt[0. 1.]  pixels=63000, 44723:18277= 2.45:1
    elif current_dataset == 'Bayarea':
        img_gt = data_preprocess.label_transform012(gt)  # Bayarea(600, 500, 224), gt[0. 1. 2.]  -> gt[2. 0. 1.]
    elif current_dataset == 'Hermiston':
        img_gt = gt  # Hermiston(307, 241, 154), gt[0. 1.]  pixels=73987, 57311:16676= 3.44:1

    # Add noise
    img1_flatten = img1.flatten(1)
    torch.manual_seed(1)
    scale = 0.01
    x1_noisy = torch.zeros_like(img1_flatten)
    for i in range(img1_flatten.shape[1]):
        noise = torch.randn([img1.shape[0]], device=img1.device) * scale
        x1_noisy[:, i] = img1_flatten[:, i] + noise
    x1_noisy = x1_noisy.reshape(img1.shape[0], img1.shape[1], img1.shape[2])

    # construct_sample
    img1_pad, img2_pad, pad_gt, x1_noisy_pad, patch_coordinates = data_preprocess.construct_sample(img1, img2, img_gt, x1_noisy, patch_size)

    # select_sample
    data_sample = data_preprocess.select_sample(img_gt, train_set_num)

    data_sample['img1'] = img1
    data_sample['img2'] = img2
    data_sample['img_gt'] = img_gt
    data_sample['img1_pad'] = img1_pad
    data_sample['img2_pad'] = img2_pad
    data_sample['pad_gt'] = pad_gt
    data_sample['x1_noisy_pad'] = x1_noisy_pad
    data_sample['patch_coordinates'] = patch_coordinates

    return data_sample
