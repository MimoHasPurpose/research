import torch.utils.data as data


class HSICD_data(data.Dataset):
    def __init__(self, data_sample, cfg):

        self.phase = cfg['phase']
        self.img1 = data_sample['img1_pad']
        self.img2 = data_sample['img2_pad']
        self.gt = data_sample['img_gt']
        self.gt_pad = data_sample['pad_gt']
        self.patch_coordinates = data_sample['patch_coordinates']
        self.x1_noisy = data_sample['x1_noisy_pad']
        if self.phase == 'trl':
            self.data_indices = data_sample['trainl_sample_center']
        elif self.phase == 'tru':
            self.data_indices = data_sample['trainu_sample_center']
        elif self.phase == 'test':
            self.data_indices = data_sample['test_sample_center']

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        index = self.data_indices[idx]
        img_index = self.patch_coordinates[index[0]]
        img1 = self.img1[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]
        img2 = self.img2[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]
        gt_2 = self.gt_pad[img_index[0]:img_index[1], img_index[2]:img_index[3]]
        x_noisy = self.x1_noisy[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]
        label_gt = self.gt[index[1], index[2]]

        return img1, img2, gt_2, x_noisy, label_gt, index
