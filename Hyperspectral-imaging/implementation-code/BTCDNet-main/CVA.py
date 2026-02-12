import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import scipy.io
from sklearn.metrics import confusion_matrix, cohen_kappa_score, recall_score, precision_score, f1_score
import time
import os

model_type = "CVA"
time_now = time.localtime()
time_folder = os.path.join("logs", time.strftime("%Y-%m-%d-%H-%M-%S", time_now)+"_"+model_type)
os.makedirs(time_folder, exist_ok=True)

tic = time.time()
file_path = r"data/Shenzhen.mat"
mat_data = scipy.io.loadmat(file_path)
# get T1 and T2 data
data_t1 = mat_data['T1']
data_t2 = mat_data['T2']
data_t1 = data_t1 / np.max(np.abs(data_t1))
data_t2 = data_t2 / np.max(np.abs(data_t2))
# get binary label
data_label = mat_data['Change'].astype(int)

# Step 1: 计算光谱变化向量
change_vector = data_t2 - data_t1

# Step 2: 计算变化幅度（欧几里得距离）
magnitude = np.sqrt(np.sum(change_vector**2, axis=-1))

# Step 3: 使用 Otsu 方法计算阈值
otsu_threshold = threshold_otsu(magnitude)
print(f"otsu_threshold: {otsu_threshold}")
toc = time.time()
print("Running Time: {:.2f}".format(toc-tic))

tic = time.time()
# Step 4: 根据阈值分割显著变化区域
change_mask = magnitude > otsu_threshold
# 将布尔类型数组转换为整数类型
change_mask = change_mask.astype(int)
change_mask[change_mask == 1] = 1
change_mask[change_mask == 0] = 2
change_mask[data_label == 0] = 0

toc = time.time()
print("Running Time: {:.2f}".format(toc-tic))
# # 可视化结果
# plt.figure(figsize=(12, 8))
# plt.subplot(1, 3, 1)
# plt.title("Change Magnitude")
# plt.imshow(magnitude, cmap="hot")
# plt.colorbar()


# # 显著变化区域
# plt.subplot(1, 3, 2)
# plt.title(f"Significant Change Mask")
# plt.imshow(change_mask, cmap="gray_r")
# plt.colorbar()

# # 显著变化区域
# plt.subplot(1, 3, 3)
# plt.title(f"Label")
# plt.imshow(data_label, cmap="gray_r")
# plt.colorbar()

# plt.tight_layout()
# plt.show()

plt.imsave(os.path.join(time_folder, "data_label.png"), data_label, cmap='gray_r', dpi=300)
plt.imsave(os.path.join(time_folder, "predict_label.png"), change_mask, cmap='gray_r', dpi=300)

data_label = data_label.flatten()
change_mask = change_mask.flatten()

# 计算混淆矩阵
change_mask[change_mask == 2] = 0
data_label[data_label == 2] = 0
conf_matrix = confusion_matrix(data_label, change_mask, labels=[0, 1])
print("Confusion Matrix:")
print(conf_matrix)

# 计算 Cohen's Kappa 系数
kappa = cohen_kappa_score(data_label, change_mask, labels=[0, 1])
print("Kappa:", kappa)

# 计算 Recall（召回率）
recall = recall_score(data_label, change_mask, average='weighted', labels=[0, 1])
print("Recall:", recall)

# 计算 Precision（精确率）
precision = precision_score(data_label, change_mask, average='weighted', labels=[0, 1])
print("Precision:", precision)

# 计算 F1 Score
f1 = f1_score(data_label, change_mask, average='weighted', labels=[0, 1])
print("F1 Score:", f1)

with open(os.path.join(time_folder, "output_metrics.txt"), 'w') as file:
    file.write("Confusion Matrix:\n")
    file.write(np.array2string(conf_matrix) + "\n\n")
    file.write(f"Cohen's Kappa: {kappa}\n")
    file.write(f"Recall (Weighted): {recall}\n")
    file.write(f"Precision (Weighted): {precision}\n")
    file.write(f"F1 Score (Weighted): {f1}\n")
