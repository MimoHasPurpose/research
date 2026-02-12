import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from fvcore.nn import FlopCountAnalysis
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

class CoAM_Module(nn.Module):
    """ Correlation attention module"""

    def __init__(self, in_dim):
        super(CoAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward (self, x1, x2):
        """
            inputs :
                x1 : input feature maps1( B X C X H X W)
                x2: input feature maps2( B X C X H X W)
            returns :
                out : attention value2 + input feature1
                attention: B X C X C
        """
        m_batchsize, C, height, width = x1.size()
        proj_query = x2.view(m_batchsize, C, -1)
        proj_key = x2.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x1.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x1
        return out
    
class deeplab_V2(nn.Module):
    def __init__(self, in_chans):
        super(deeplab_V2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),

        )
        '''
        '''
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),#![](classification_maps/IN_gt.png)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),

        )  #进行卷积操作


        inter_channels = 512 // 4###################################################
        self.conv5a = nn.Sequential(nn.Conv2d(512, inter_channels, 3, padding=1, bias=False),

                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(512, inter_channels, 3, padding=1, bias=False),

                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)####
        self.sc = CAM_Module(inter_channels)
        self.sco = CoAM_Module(inter_channels)###
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),

                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),

                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 128, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 128, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 128, 1))
        self.conv9 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 128, 1))

        ####### multi-scale contexts #######
        ####### dialtion = 6 ##########
        self.fc6_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=6, padding=6),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        ####### dialtion = 12 ##########
        self.fc6_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=12, padding=12),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.fc7_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        ####### dialtion = 18 ##########
        self.fc6_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=18, padding=18),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        ####### dialtion = 24 ##########
        self.fc6_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=24, padding=24),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.embedding_layer = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        # self.fc8 = nn.Softmax2d()
        # self.fc8 = fun.l2normalization(scale=1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        feat11 = self.conv5a(x1)
        sa_feat1 = self.sa(feat11)
        sa_conv1 = self.conv51(sa_feat1)
        sa_output1 = self.conv6(sa_conv1)

        feat12 = self.conv5c(x1)
        sc_feat1 = self.sc(feat12)
        sc_conv1 = self.conv52(sc_feat1)
        sc_output = self.conv7(sc_conv1)

        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        feat21 = self.conv5a(x2)
        sa_feat2 = self.sa(feat21)
        sa_conv2 = self.conv51(sa_feat2)
        sa_output2 = self.conv6(sa_conv2)

        feat22 = self.conv5c(x2)
        sc_feat2 = self.sc(feat22)
        sc_conv2 = self.conv52(sc_feat2)
        sc_output = self.conv7(sc_conv2)

        sco_conv1 = self.sco(feat11,feat12)
        sco_conv1 = self.conv51(sco_conv1)
        sco_conv2 = self.sco(feat12,feat11)
        sco_conv2 = self.conv51(sco_conv2)

        feat_sum1 = sa_conv1 + sc_conv1 + 0.3*sco_conv1
        feat_sum2 = sa_conv2 + sc_conv2 + 0.3*sco_conv2
        sasc_output1 = self.conv8(feat_sum1)
        sasc_output2 = self.conv8(feat_sum2)


        return sasc_output1, sasc_output2


class SiameseNet(nn.Module):
    def __init__(self, in_chans, norm_flag='l2'):
        super(SiameseNet, self).__init__()
        self.CNN = deeplab_V2(in_chans)
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 128, 1))
        self.conv9 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 128, 1))


        if norm_flag == 'l2':
            self.norm = F.normalize  #F.normalize对输入的数据（tensor）进行指定维度的L2_norm运算
        if norm_flag == 'exp':
            self.norm = nn.Softmax2d()

    def forward(self, t0, t1):
        t0 = t0.float()
        t1 = t1.float()
        out_t0_embedding, out_t1_embedding, = self.CNN(t0, t1)
        return out_t0_embedding, out_t1_embedding


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, 3, padding=1, padding_mode='reflect', bias=False)  # [64, 24, 24]
        self.bat1 = nn.BatchNorm2d(64)#
        self.reli1 = nn.LeakyReLU(0.2)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(64, 32, 3, padding=1, padding_mode='reflect', bias=False)
        self.bat2 = nn.BatchNorm2d(32)
        self.reli2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(64, 256, 3, padding=1, padding_mode='reflect', bias=False)
        self.bat3 = nn.BatchNorm2d(256)
        self.reli3 = nn.LeakyReLU(0.2)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):
        con1 = self.conv1(x)
        ba1 = self.bat1(con1)
        re1 = self.reli1(ba1)
        po1 = self.pool1(re1)
        con2 = self.conv2(po1)
        ba2 = self.bat2(con2)
        re2 = self.reli2(ba2)

        return re2


class ChangeNet(nn.Module):
    def __init__(self):
        super(ChangeNet, self).__init__()
        self.singlebrach = Classifier()# re2
        self.fc = nn.Sequential(        #一个有序的容器
            nn.Linear(32, 16),#32和16是维度
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, t0, t1):

        indata = t0 - t1
        c3 = self.singlebrach(indata)

        return c3


class CSANet(nn.Module):#######################nn,Module################################################################
    def __init__(self, in_chans, num_classes):
        super(CSANet, self).__init__()
        self.siamesnet = SiameseNet(in_chans)
        self.chnet = ChangeNet() #c3
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, t0, t1):
        t0 = t0.permute(0, 3, 1, 2) #换个顺序 0123-----0312
        t1 = t1.permute(0, 3, 1, 2)

        x1, x2 = self.siamesnet(t0, t1)
        out = self.chnet(x1, x2)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.sigmoid(out)


        return out
    

if __name__ == "__main__":
    # 创建模型实例，定义输入参数
    band = 166
    num_classes = 3

    # 测试推理时间
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CSANet(in_chans=band, num_classes=num_classes).to(device)

    # 打印模型结构（可选）
    print(model)

    # 随机生成一个输入，假设输入尺寸为 (B, C, H, W) = (1, 3, 224, 224)
    input1 = torch.randn(2, 9, 9, 166).to(device)
    input2 = torch.randn(2, 9, 9, 166).to(device)

    # 运行模型并打印输出形状
    output = model(input1, input2)
    print(output.shape)

    # GPU 计时
    if device == 'cuda':
        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        with torch.no_grad():
            _ = model(input1, input2)
        ender.record()
        torch.cuda.synchronize()
        print(f"Inference time: {starter.elapsed_time(ender)} ms")

    # CPU 计时
    else:
        start_time = time.time()
        with torch.no_grad():
            _ = model(input1, input2)
        end_time = time.time()
        print(f"Inference time: {(end_time - start_time) * 1000} ms")

    # 计算并打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6}M")
    print(f"Trainable parameters: {trainable_params/1e6}M")

    flops = FlopCountAnalysis(model, (input1, input2))
    print(f"FLOPs: {flops.total() / 1e6} MFLOPs")  # 将FLOPs转换为GFLOPs