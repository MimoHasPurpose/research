import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from fvcore.nn import FlopCountAnalysis
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class GETNET(nn.Module):
    def __init__(self, in_chans, num_classes):
        super(GETNET, self).__init__()
        
        # LSConv1 layer
        self.lsconv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.Tanh()
        )

        # LSConv2 layer
        self.lsconv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh()
        )

        # LSConv3 layer
        self.lsconv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh()
        )

        # LSConv4 layer
        self.lsconv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=1, stride=1),
            nn.BatchNorm2d(96),
            nn.Tanh()
        )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 512),  # W和H分别经过4次 2x2 池化
            nn.BatchNorm1d(512),
            nn.Tanh(),
        )
        self.fc2 = nn.Linear(512, num_classes) 

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2):
        x1 = x1.permute(0, 3, 1, 2) 
        x2 = x2.permute(0, 3, 1, 2)

        x1 = self.lsconv1(x1)
        x2 = self.lsconv1(x2)

        x1 = self.maxpool2(self.lsconv2(x1))
        x2 = self.maxpool2(self.lsconv2(x2))

        x1 = self.lsconv3(x1)
        x2 = self.lsconv3(x2)

        x1 = self.maxpool4(self.lsconv4(x1))
        x2 = self.maxpool4(self.lsconv4(x2))

        x1 = self.global_avg_pool(x1)
        x2 = self.global_avg_pool(x2)
        
        x1 = x1.view(x1.size(0), -1)  # Flatten
        x2 = x2.view(x2.size(0), -1)  # Flatten

        x = self.fc1(x1+x2)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # 创建模型实例，定义输入参数
    num_classes = 3
    band = 166

    # 测试推理时间
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GETNET(in_chans=band, num_classes=num_classes).to(device)

    # 打印模型结构（可选）
    print(model)

    # 随机生成一个输入，假设输入尺寸为 (B, C, H, W) = (1, 3, 224, 224)
    input1 = torch.randn(2, 5, 5, 166).to(device)
    input2 = torch.randn(2, 5, 5, 166).to(device)

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