# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Convblock(nn.Module):
    def __init__(self, C_in, C_out):
        super(Convblock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


def conv3(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UpSampling(nn.Module):
    def __init__(self, C):
        super(UpSampling, self).__init__()
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LMM(nn.Module):
    def __init__(self, channels):
        super(LMM, self).__init__()
        self.channels = channels
        dim = self.channels
        # 3*7conv
        self.fc_h = nn.Conv2d(dim, dim, (3, 7), stride=1, padding=(1, 7 // 2), groups=dim, bias=False)
        self.fc_w = nn.Conv2d(dim, dim, (7, 3), stride=1, padding=(7 // 2, 1), groups=dim, bias=False)
        self.reweight = Mlp(dim, dim // 2, dim * 3)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        N, C, H, W = x.shape
        x_w = self.fc_h(x)
        x_h = self.fc_w(x)
        x_add = x_h + x_w + x
        att = F.adaptive_avg_pool2d(x_add, output_size=1)
        att = self.reweight(att).reshape(N, C, 3).permute(2, 0, 1)
        att = self.swish(att).unsqueeze(-1).unsqueeze(-1)
        x_att = x_h * att[0] + x_w * att[1] + x * att[2]

        return x_att


class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)
        trunc_normal_(self.embeddings_table, std=.02)

    def forward(self, length_q, length_k):
        # H, W
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings


class GMM(nn.Module):
    def __init__(self, channels, H, W):
        super(GMM, self).__init__()
        self.channels = channels
        patch = 4
        self.C = int(channels / patch)
        self.proj_h = nn.Conv2d(H * self.C, self.C * H, (3, 3), stride=1, padding=(1, 1), groups=self.C, bias=True)
        self.proj_w = nn.Conv2d(W * self.C, self.C * W, (3, 3), stride=1, padding=(1, 1), groups=self.C, bias=True)

        self.fuse_h = nn.Conv2d(channels * 2, channels, (1, 1), (1, 1), bias=False)
        self.fuse_w = nn.Conv2d(channels * 2, channels, (1, 1), (1, 1), bias=False)

        self.relate_pos_h = RelativePosition(channels, H)
        self.relate_pos_w = RelativePosition(channels, W)
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(channels)

    def forward(self, x):
        N, C, H, W = x.shape
        pos_h = self.relate_pos_h(H, W).unsqueeze(0).permute(0, 3, 1, 2)
        pos_w = self.relate_pos_w(H, W).unsqueeze(0).permute(0, 3, 1, 2)
        C1 = int(C / self.C)

        x_h = x + pos_h
        # Splitting & Concatenate
        x_h = x_h.view(N, C1, self.C, H, W)
        # Column
        x_h = x_h.permute(0, 1, 3, 2, 4).contiguous().view(N, C1, H, self.C * W)
        x_h = self.proj_h(x_h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_h = x_h.view(N, C1, H, self.C, W).permute(0, 1, 3, 2, 4).contiguous().view(N, C, H, W)
        x_h = self.fuse_h(torch.cat([x_h, x], dim=1))
        x_h = self.activation(self.BN(x_h)) + pos_w
        # Row
        x_w = self.proj_w(x_h.view(N, C1, H * self.C, W).permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_w = x_w.contiguous().view(N, C1, self.C, H, W).view(N, C, H, W)
        x = self.fuse_w(torch.cat([x, x_w], dim=1))

        return x


class UnetGLencoder(nn.Module):
    def __init__(self):
        super(UnetGLencoder, self).__init__()
        self.D1 = DownSampling(128)
        self.C1 = Convblock(128, 256)
        self.D2 = DownSampling(256)
        self.C2 = Convblock(256, 512)
        self.D3 = DownSampling(512)
        self.C3 = Convblock(512, 1024)

        self.L1 = LMM(128)
        self.L2 = LMM(256)
        self.L3 = LMM(512)

        self.G1 = GMM(128, 32, 32)
        self.G2 = GMM(256, 16, 16)
        self.G3 = GMM(512, 8, 8)

    def forward(self, x):
        # unet + GMM +LMM
        x1 = self.L1(self.G1(x))
        R2 = self.C1(self.D1(x1))
        x2 = self.L2(self.G2(R2))
        R3 = self.C2(self.D2(x2))
        x3 = self.L3(self.G3(R3))
        Y = self.C3(self.D3(x3))

        return x, R2, R3, Y


class Sub_UNETdecoder(nn.Module):
    def __init__(self, in_fea_num):
        super(Sub_UNETdecoder, self).__init__()
        self.U1 = UpSampling(1024)
        self.C1 = Convblock(1024, 512)
        self.U2 = UpSampling(512)
        self.C2 = Convblock(512, 256)
        self.U3 = UpSampling(256)
        self.C3 = Convblock(256, 128)

    def forward(self, x, R2, R3, Y):
        O1 = self.C1(torch.cat((self.U1(Y), R3), 1))
        O2 = self.C2(torch.cat((self.U2(O1), R2), 1))
        out = self.C3(torch.cat((self.U3(O2), x), 1))

        return out
