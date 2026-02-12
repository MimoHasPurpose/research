# -*- coding: utf-8 -*-
import random
from model.mask_utils import *
from model.ecodec import *


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def Mask_Generate(x):
    B, C, h, w = x.shape
    mask_size_choice = [2, 4, 8]
    probabilities_mask = [0.333, 0.334, 0.333]
    mask_patch_size = random_pick(mask_size_choice, probabilities_mask)
    mask_ratio = 0.6  # F , H   , if Bay : mask_ratio = 0.8
    mask = Mask_PerPatch_rectangle(x, mask_patch_size=mask_patch_size, mask_ratio=mask_ratio)
    mask = mask.unsqueeze(0).expand(B, h, w)

    return mask


class stmnet(nn.Module):
    '''Single-temporal Mask based Network'''

    def __init__(self, in_fea_num, feature_num=128):
        super(stmnet, self).__init__()
        # data
        self.init = nn.Conv2d(in_fea_num, feature_num, kernel_size=1)
        self.addmask = AddMaskImgDiff(embed_dim=128)

        # ecodec
        self.encoder = UnetGLencoder()
        self.decoder_d = Sub_UNETdecoder(in_fea_num)
        self.decoder_r = Sub_UNETdecoder(in_fea_num)

        # output
        self.fc = nn.Conv2d(128, 2, kernel_size=1)
        self.mlp_r = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 128, 1),
        )
        self.mlp_d = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 128, 1),
        )

    def forward(self, img1, x_noisy):
        x = self.init(img1)
        x_noisy = self.init(x_noisy)

        # add mask
        mask_now = Mask_Generate(img1)
        x_mask, gt_mask = self.addmask(x, mask_now, x_noisy)

        # encoder
        x_d1, x_d2, x_d3, x_d4 = self.encoder(x)
        xm_d1, xm_d2, xm_d3, xm_d4 = self.encoder(x_mask)

        # Decoder
        d1, d2, d3, d4 = x_d1 - xm_d1, x_d2 - xm_d2, x_d3 - xm_d3, x_d4 - xm_d4
        x_detect = self.decoder_d(d1, d2, d3, d4)
        x_rec = self.decoder_r(xm_d1, xm_d2, xm_d3, xm_d4)

        # mlp
        x_detect = self.mlp_d(x_detect)
        x_rec = self.mlp_r(x_rec)
        x_detect = self.fc(x_detect)

        gt_mask = gt_mask.clone().detach().cuda()
        loss_detect = F.cross_entropy(x_detect, gt_mask.long(), reduction='mean')  #
        loss_recon = F.l1_loss(x_rec, x.long(), reduction='mean')  #

        loss = loss_recon + loss_detect

        return x_detect, x_rec, loss
