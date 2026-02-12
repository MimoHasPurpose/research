# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from timm.models.layers import trunc_normal_

'''Patch Cut&Reduction'''
class PatchCut_overlap(nn.Module):
    def __init__(self, H, W, patch_size=32, overlap_factor=2):
        super(PatchCut_overlap, self).__init__()
        self.H = H
        self.W = W
        self.patch_size = patch_size
        self.stride = patch_size - overlap_factor

        pad_h = self.patch_size - self.H % self.stride
        pad_w = self.patch_size - self.W % self.stride
        self.pad = nn.ReflectionPad2d(padding=(0, pad_w, 0, pad_h))

    def forward(self, x):
        pad = self.pad(x.unsqueeze(0))
        N, C, H_pad, W_pad = pad.shape
        index = 0
        patch_cluster = torch.zeros(([1, C, self.patch_size, self.patch_size])).to('cuda')

        for i in range(0, H_pad // self.stride):
            for j in range(0, W_pad // self.stride):
                index = index + 1
                topleft_y = i * self.stride
                topleft_x = j * self.stride
                img_cut = pad[:, :, topleft_y:topleft_y + self.patch_size, topleft_x:topleft_x + self.patch_size]
                if index == 1:
                    patch_cluster = img_cut
                else:
                    patch_cluster = torch.cat([patch_cluster, img_cut], dim=0)

        return patch_cluster


class PatchCut_gt(nn.Module):
    def __init__(self, H, W, patch_size=224):
        super(PatchCut_gt, self).__init__()
        self.H = H
        self.W = W
        self.patch_size = patch_size
        self.stride = patch_size

        pad_h = self.patch_size - self.H % self.patch_size
        pad_w = self.patch_size - self.W % self.patch_size
        self.pad = nn.ReflectionPad2d(padding=(0, pad_w, 0, pad_h))  # 左右上下

    def forward(self, x):
        pad = self.pad(x.unsqueeze(0))  # torch.Size([1, 128, 672, 224])
        N, H_pad, W_pad = pad.shape
        assert H_pad % self.patch_size == 0
        assert W_pad % self.patch_size == 0

        index = 0
        patch_cluster = torch.zeros(([1, self.patch_size, self.patch_size])).to('cuda')

        for i in range(0, H_pad // self.stride):
            for j in range(0, W_pad // self.stride):
                index = index + 1
                topleft_y = i * self.stride
                topleft_x = j * self.stride
                img_cut = pad[:, topleft_y:topleft_y + self.patch_size, topleft_x:topleft_x + self.patch_size]
                if index == 1:
                    patch_cluster = img_cut
                else:
                    patch_cluster = torch.cat([patch_cluster, img_cut], dim=0)
        return patch_cluster


class PatchReduction_overlap(nn.Module):
    def __init__(self, H, W, patch_size=128, overlap_factor=2):
        super(PatchReduction_overlap, self).__init__()
        self.H = H
        self.W = W
        self.patch_size = patch_size
        self.stride = patch_size - overlap_factor
        self.H_pad = self.H + self.patch_size - self.H % self.stride
        self.W_pad = self.W + self.patch_size - self.W % self.stride

    def forward(self, x):
        _, C, _, _ = x.shape
        img_origin_pad = torch.zeros(([C, self.H_pad, self.W_pad])).to('cuda')
        index = 0
        for i in range(0, self.H_pad // self.stride):
            for j in range(0, self.W_pad // self.stride):
                topleft_y = i * self.stride
                topleft_x = j * self.stride
                img_origin_pad[:, topleft_y:topleft_y + self.patch_size, topleft_x:topleft_x + self.patch_size] = x[
                    index]
                index = index + 1
        img_origin = img_origin_pad[:, : self.H, : self.W]  # torch.Size([2, 450, 140])

        return img_origin


'''Masking Strategy'''
def Mask_PerPatch_rectangle(img, mask_patch_size=8, mask_ratio=0.6):
    B, C, H, W = img.shape
    mask_HW = torch.zeros([H, W])

    rand_size_h = H // mask_patch_size
    rand_size_w = W // mask_patch_size
    token_count = rand_size_h * rand_size_w
    mask_count = int(np.ceil(token_count * mask_ratio))

    mask_idx = np.random.permutation(token_count)[:mask_count]
    mask = np.zeros(token_count, dtype=int)
    mask[mask_idx] = 1

    # Mask block filling
    mask = mask.reshape((rand_size_h, rand_size_w))
    mask = mask.repeat(mask_patch_size, axis=0).repeat(mask_patch_size, axis=1)
    mask = torch.from_numpy(mask)
    mask_h, mask_w = mask.shape
    mask_HW[0:mask_h, 0:mask_w] = mask

    return mask_HW


class AddMaskImgDiff(nn.Module):
    def __init__(self, embed_dim=128):
        super(AddMaskImgDiff, self).__init__()
        # learnable masked tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed_dim = embed_dim
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask, x_noisy):
        assert mask is not None
        B, C, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # B h*w C
        x_noisy = x_noisy.flatten(2).transpose(1, 2)  # B h*w C
        B, L, dim = x.shape

        # Add Mask
        mask_tokens = self.mask_token.expand(B, L, -1)
        mweight_ = mask.flatten(1).type_as(mask_tokens)  #
        mweight = mweight_.clone().detach().unsqueeze(2)
        # Spectral flipping
        x_change = torch.flip(x[:, h * w // 2, :], dims=[-1]).unsqueeze(1)
        x_mask = torch.mul(x_noisy, (1. - mweight)) + torch.mul(x_change, mweight)

        x_mask = rearrange(x_mask, 'b (h w) c -> b c h w', h=h, w=w)
        gt_mask = mask

        return x_mask, gt_mask
