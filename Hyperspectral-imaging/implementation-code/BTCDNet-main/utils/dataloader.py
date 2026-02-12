import os 
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class HSI_Dataset(Dataset):
    def __init__(self, data_t1, data_t2, labels, patch_size=5, band_patches=3, mode="train"):
        """
        初始化数据集
        :param data_t1: 高光谱图像数据 t1
        :param data_t2: 高光谱图像数据 t2
        :param labels: 对应的标签
        :param patch_size: 每个小块的大小
        :param band_patch: 增强的波段数量
        """
        self.data_t1 = self.normalize_data(data_t1)
        self.data_t2 = self.normalize_data(data_t2)
        self.labels = labels
        self.patch_size = patch_size
        self.band_patch = band_patches
        self.band_patch = band_patches
        self.mode = mode
        self.height, self.width, self.band = self.data_t1.shape

        # 使用镜像填充边界
        self.padded_data_t1 = self.pad_data(self.data_t1, "data")
        self.padded_data_t2 = self.pad_data(self.data_t2, "data")
        self.padded_labels = self.pad_data(self.labels, "label")

        # 记录原始标签中有效的索引
        self.valid_indices = self.get_valid_indices()

    def normalize_data(self, data):
        """
        对输入数据进行归一化到 [0, 1] 或标准化
        :param data: 高光谱图像数据
        :return: 归一化后的数据
        """
        # # 最小-最大归一化
        # min_val = np.min(data)
        # max_val = np.max(data)
        # normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
        
        # # 如果需要标准化，可以用以下代码
        mean = np.mean(data, axis=(0, 1))  # 每个波段的均值
        std = np.std(data, axis=(0, 1))   # 每个波段的标准差
        normalized_data = (data - mean) / (std + 1e-8)

        return normalized_data
    
    def pad_data(self, data, mode):
        """
        对数据进行镜像填充
        :param data: 输入数据
        :return: 填充后的数据
        """
        pad_width = self.patch_size // 2
        # 确保填充为 (上下, 左右) 的格式
        if mode == "data":
            padded_data = np.pad(data, 
                                ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), 
                                mode='reflect')  # 使用反射模式填充
        else:
            padded_data = np.pad(data, 
                                ((pad_width, pad_width), (pad_width, pad_width)), 
                                mode='reflect')  # 使用反射模式填充
        return padded_data

    def get_valid_indices(self):
        """
        获取有效的标签索引，确保不为0的原始标签被纳入有效索引
        :return: 有效标签的索引
        """
        pad_width = self.patch_size // 2

        if self.mode == "train":
            padded_label = np.pad(self.labels, 
                                ((pad_width, pad_width), (pad_width, pad_width)), 
                                mode='constant', constant_values=0)  # 使用常量填充
            valid_indices = np.argwhere(padded_label > 0)
        elif self.mode == "test":
            padded_label = np.pad(self.labels, 
                                ((pad_width, pad_width), (pad_width, pad_width)), 
                                mode='constant', constant_values=-1)  # 使用常量填充
            valid_indices = np.argwhere(padded_label > -1)
        else:
            raise ValueError("mode error!")

        return valid_indices

    def __len__(self):
        """
        返回数据集的大小
        :return: 样本数量
        """
        return len(self.valid_indices)  # 只返回有效样本的数量

    def __getitem__(self, idx):
        """
        获取指定索引的样本
        :param idx: 索引
        :return: (data_t1, data_t2, 标签)
        """
        # 获取有效像素的行列索引
        row, col = self.valid_indices[idx]

        # 提取小块的边界
        half_patch = self.patch_size // 2
        start_row = row - half_patch
        end_row = row + half_patch + 1
        start_col = col - half_patch
        end_col = col + half_patch + 1

        # 从填充后的数据中提取 patch
        data_t1 = self.padded_data_t1[start_row:end_row, start_col:end_col, :]
        data_t2 = self.padded_data_t2[start_row:end_row, start_col:end_col, :]
        
        # 获取对应标签
        label = self.padded_labels[row, col]

        if self.band_patch >= 3:
            # 增强波段的处理
            data_t1_enhanced = self.gain_neighborhood_band(data_t1)
            data_t2_enhanced = self.gain_neighborhood_band(data_t2)
        else:
            data_t1_enhanced = data_t1
            data_t2_enhanced = data_t2

        # 转换为张量
        data_t1 = torch.tensor(data_t1_enhanced, dtype=torch.float32)
        data_t2 = torch.tensor(data_t2_enhanced, dtype=torch.float32)
        label = torch.tensor(label)

        return data_t1, data_t2, label

    def gain_neighborhood_band(self, x):
        """
        根据 band_patch 增强波段
        :param x_train: 输入数据
        :return: 增强后的数据
        """
        nn = self.band_patch // 2
        pp = (self.patch_size * self.patch_size) // 2
        x_reshape = x.reshape(self.patch_size * self.patch_size, self.band)
        
        # 创建增强波段的数组
        x_band = np.zeros((self.patch_size * self.patch_size * self.band_patch, self.band), dtype=float)
        
        # 中心区域
        x_band[nn * self.patch_size * self.patch_size:(nn + 1) * self.patch_size * self.patch_size, :] = x_reshape
        
        # 左边镜像
        for i in range(nn):
            if pp > 0:
                x_band[i * self.patch_size * self.patch_size:(i + 1) * self.patch_size * self.patch_size, :i + 1] = x_reshape[:, self.band - i - 1:]
                x_band[i * self.patch_size * self.patch_size:(i + 1) * self.patch_size * self.patch_size, i + 1:] = x_reshape[:, :self.band - i - 1]
            else:
                x_band[i:(i + 1), :(nn - i)] = x_reshape[0:1, (self.band - nn + i):]
                x_band[i:(i + 1), (nn - i):] = x_reshape[0:1, :(self.band - nn + i)]
        
        # 右边镜像
        for i in range(nn):
            if pp > 0:
                x_band[(nn + i + 1) * self.patch_size * self.patch_size:(nn + i + 2) * self.patch_size * self.patch_size, :self.band - i - 1] = x_reshape[:, i + 1:]
                x_band[(nn + i + 1) * self.patch_size * self.patch_size:(nn + i + 2) * self.patch_size * self.patch_size, self.band - i - 1:] = x_reshape[:, :i + 1]
            else:
                x_band[(nn + 1 + i):(nn + 2 + i), (self.band - i - 1):] = x_reshape[0:1, :(i + 1)]
                x_band[(nn + 1 + i):(nn + 2 + i), :(self.band - i - 1)] = x_reshape[0:1, (i + 1):]
        
        return x_band

# 使用示例
if __name__ == "__main__":
    # 这里假设你已经加载了 data_t1, data_t2, 和 labels
    data_t1 = np.random.rand(100, 100, 166)  # 示例数据，100x100 像素，166 个波段
    data_t2 = np.random.rand(100, 100, 166)  # 示例数据
    labels = np.random.randint(0, 2, (100, 100))  # 示例标签，二值化（0 和 1）

    patch_size = 5  # 每个小块的大小
    band_patches = 1  # 增强的波段数量

    # 创建数据集实例
    hsi_dataset = HSI_Dataset(data_t1, data_t2, labels, patch_size, band_patches, "test")
    hsi_dataset = HSI_Dataset(data_t1, data_t2, labels, patch_size, band_patches, "test")

    # 打印数据集的大小
    print(f"Dataset Size: {len(hsi_dataset)}")

    # 打印数据块的大小
    data_t1_sample, data_t2_sample, label_sample = hsi_dataset[0]
    print(f"data_t1 sample size: {data_t1_sample.shape}")
    print(f"data_t2 sample size: {data_t2_sample.shape}")
    print(f"label sample size: {label_sample.shape}")
