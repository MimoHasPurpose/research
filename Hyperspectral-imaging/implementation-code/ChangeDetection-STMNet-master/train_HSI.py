# -*- coding: utf-8 -*-
import torch
torch.cuda.set_device(0)
import torch.optim as optim
import imageio
from scipy import io

import configs.configs as cfg
from data.get_train_test_set import get_tratest_set_batch as get_set_b
from data.HSICD_data import HSICD_data
from tools.train_utils import *
from tools.test_utils import test_batch_full as test_batch_full
from tools.assessment import *
from model.mainmodel import *


def main():
    current_dataset = cfg.current_dataset
    current_model = cfg.current_model
    model_name = current_dataset + current_model

    cfg_data = cfg.data
    cfg_train_pre = cfg.train['train_pre_model']
    cfg_optim_pre = cfg.train['optimizer_pre']
    cfg_test = cfg.test
    in_fea_num, H, W = cfg.model[current_dataset]

    save_folder = cfg_train_pre['save_folder']
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # 递归创建目录
    logger = gen_log(save_folder + model_name)
    logger.info('model_name:{}'.format(model_name))

    data_sets = get_set_b(cfg_data, logger)
    img_gt = data_sets['img_gt']
    img1 = data_sets['img1']
    img2 = data_sets['img2']
    trl_data = HSICD_data(data_sets, cfg_data['trl_data'])
    tru_data = HSICD_data(data_sets, cfg_data['tru_data'])
    test_data = HSICD_data(data_sets, cfg_data['test_data'])

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = stmnet(in_fea_num, feature_num=128).to(device)
    optimizer_pre = optim.SGD(model.parameters(), lr=cfg_optim_pre['lr'], momentum=cfg_optim_pre['momentum'],weight_decay=cfg_optim_pre['weight_decay'])

    # train
    train_pre_batch(tru_data, model, optimizer_pre, device, cfg_train_pre, logger)
    # test
    predict_img = test_batch_full(img1, img2, img_gt, model, device, cfg_test)

    # 精度评价
    if current_dataset == 'Bayarea':
        conf_mat, oa, kappa_co, P, R, F1, acc, oa_0, oa_1 = aa_012(img_gt, predict_img)
    else:
        conf_mat, oa, kappa_co, P, R, F1, acc, oa_0, oa_1 = accuracy_assessment(img_gt, predict_img)
    assessment_pre = [round(oa, 4) * 100, round(kappa_co, 4), round(F1, 4) * 100, round(P, 4) * 100, round(R, 4) * 100, round(oa_0, 4) * 100, round(oa_1, 4) * 100]
    logger.info('model_name: {} assessment_pre：oa, kappa_co, F1, P, R, oa_0, oa_1 || {}'.format(model_name, assessment_pre))

    # save
    save_folder = cfg_test['save_folder']
    save_name = cfg_test['save_name']
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    io.savemat(save_folder + '/' + save_name + ".mat",  {"predict_img": np.array(predict_img.cpu()), "oa_final": assessment_pre})
    predict_img = np.array(predict_img * 255, dtype=np.uint8)
    imageio.imwrite(save_folder + '/' + save_name + '+predict_img.png', predict_img)
    print('save predict_img successful!')


if __name__ == '__main__':
    main()

