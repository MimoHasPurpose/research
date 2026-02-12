import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from fvcore.nn import FlopCountAnalysis
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class DynamicDilatedConvBranch(nn.Module):
    def __init__(self, in_channels):
        super(DynamicDilatedConvBranch, self).__init__()
        self.in_channels = in_channels

        # 第一层卷积：固定 dilation=2，动态调整 stride 和 padding
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, dilation=2)
        # 第二层卷积：固定 dilation=2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=2)
        # 第三层卷积：动态调整 kernel_size, padding=0，确保输出为 1x1
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x)) 
        x = F.relu(self.conv3(x))
        x = self.global_avg_pool(x) 
        return x

class ReCNN(nn.Module):
    def __init__(self, in_chans=3, lstm_hidden_size=128, num_classes=3):
        super(ReCNN, self).__init__()
        self.T1_branch = DynamicDilatedConvBranch(in_chans)
        self.T2_branch = DynamicDilatedConvBranch(in_chans)

        # LSTM处理时间序列特征
        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden_size, batch_first=True)

        # 全连接分类层
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x1, x2):
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)

        # 分别通过 T1 和 T2 的卷积分支
        T1_features = self.T1_branch(x1)  # [batch_size, 256, 1, 1]
        T2_features = self.T2_branch(x2)  # [batch_size, 256, 1, 1]

        # 将特征压缩为 (batch_size, feature_dim)
        T1_features = T1_features.squeeze(-1).squeeze(-1)  # [batch_size, 256]
        T2_features = T2_features.squeeze(-1).squeeze(-1)  # [batch_size, 256]

        # 将 T1 和 T2 特征堆叠为时间序列
        seq_features = torch.stack([T1_features, T2_features], dim=1)  # [batch_size, seq_len, feature_dim]

        # 通过 LSTM
        lstm_out, _ = self.lstm(seq_features)  # [batch_size, seq_len, lstm_hidden_size]
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出

        # 分类
        out = self.fc(lstm_out)  # [batch_size, num_classes]
        return out
    
if __name__ == "__main__":
    # 创建模型实例，定义输入参数
    band = 166
    num_classes = 3

    # 测试推理时间
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ReCNN(in_chans=band, num_classes=num_classes).to(device)

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