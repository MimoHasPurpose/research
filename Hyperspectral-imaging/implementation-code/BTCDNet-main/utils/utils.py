import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# a class for calculating the average of the accuracy and the loss
#-------------------------------------------------------------------------------
class AverageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.average = 0 
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.count += n
    self.average = self.sum / self.count
#-------------------------------------------------------------------------------

    
def load_hsi_mat_data(file_path, num_samples):
    # load mat data
    mat_data = scipy.io.loadmat(file_path)

    # get T1 and T2 data
    data_t1 = mat_data['T1']
    data_t2 = mat_data['T2']
    
    # get binary label
    data_label = mat_data['Change'].astype(int)
    
    # search negative and positive location
    negative_position = np.array(np.where(data_label == 2)).transpose(1, 0)
    positive_position = np.array(np.where(data_label == 1)).transpose(1, 0)

    # tandom select data
    selected_negative = np.random.choice(negative_position.shape[0], int(num_samples), replace=False)
    selected_positive = np.random.choice(positive_position.shape[0], int(num_samples), replace=False)
    
    selected_negative_position = negative_position[selected_negative]
    selected_positive_position = positive_position[selected_positive]
    
    # initialize train data
    train_label = np.zeros(data_label.shape)
    for i in range(num_samples):
        train_label[selected_positive_position[i][0], selected_positive_position[i][1]] = 1  # 1-positive
        train_label[selected_negative_position[i][0], selected_negative_position[i][1]] = 2  # 2-negative

    # initialize test data
    test_label = data_label - train_label  # remove train_label from data_label

    return data_t1, data_t2, train_label, test_label, data_label, negative_position.shape[0], positive_position.shape[0]

#-------------------------------------------------------------------------------

def save_labels_images(train_label, test_label, output_dir):
    # 保存 train_label 的图像
    plt.imshow(train_label, cmap='gray')
    plt.title('Training Labels (TR)')
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(f"{output_dir}/train_label.png", bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭当前图形

    # 保存 test_label 的图像
    plt.imshow(test_label, cmap='gray')
    plt.title('Test Samples (TE)')
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(f"{output_dir}/test_label.png", bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭当前图形