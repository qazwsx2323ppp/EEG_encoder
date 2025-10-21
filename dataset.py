# my_dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import os


class TripletDataset(Dataset):
    """
    用于加载 (EEG, 图像向量, 文本向量) 三元组的数据集。

    它假定：
    1. eeg_path 指向一个 .pth 文件，该文件包含一个列表或张量，
       其中存储了2000个项目（对应2000张图片）。
    2. 每个项目包含6个受试者的EEG数据，形状为 (6, 128, 440)
       (受试者, 通道, 采样点)。
    3. image_vec_path 和 text_vec_path 指向 .npy 文件，
       形状为 (2000, embedding_dim)，顺序与EEG数据一致。
    4. splits_path 指向 'block_splits_by_image_single.pth' 文件，
       其中的索引 (0-1999) 用于划分2000张图片。
    """

    def __init__(self, cfg_data, mode='train'):
        print(f"正在加载 {mode} 数据...")

        # 1. 加载所有数据到内存
        #    我们假设EEG数据文件加载后是一个列表或字典
        #    为了简化，我们假设'dataset'键下是一个列表，包含2000个元素
        #    请根据您的 .pth 文件实际结构调整 'dataset' 这个键
        try:
            self.all_eeg_data = torch.load(cfg_data.eeg_path)['dataset']
        except Exception as e:
            print(f"警告：无法按预期加载EEG数据（{e}）。尝试直接加载。")
            self.all_eeg_data = torch.load(cfg_data.eeg_path)

        self.all_image_vectors = torch.from_numpy(np.load(cfg_data.image_vec_path)).float()
        self.all_text_vectors = torch.from_numpy(np.load(cfg_data.text_vec_path)).float()

        # 2. 加载数据划分索引
        splits_data = torch.load(cfg_data.splits_path)
        self.indices = splits_data['splits'][mode]  # e.g., 'train', 'val', 'test'

        print(f"加载了 {len(self.all_image_vectors)} 个总样本。")
        print(f"为 {mode} 模式找到了 {len(self.indices)} 个索引。")

    def __len__(self):
        # 返回当前模式（train/val/test）下的样本数量
        return len(self.indices)

    def __getitem__(self, idx):
        # 1. 根据传入的idx，从当前模式的索引列表中获取“主索引”
        main_index = self.indices[idx]

        # 2. 使用主索引从所有数据中提取对应的三元组

        # 加载EEG数据，并随机选择一个受试者的数据
        # 假设数据形状为 (num_subjects, num_channels, num_samples)
        eeg_all_subjects = self.all_eeg_data[main_index]
        num_subjects = eeg_all_subjects.shape[0]
        subject_idx = torch.randint(0, num_subjects, (1,)).item()
        eeg_signal = eeg_all_subjects[subject_idx].float()

        # 获取预先计算好的向量
        image_vector = self.all_image_vectors[main_index]
        text_vector = self.all_text_vectors[main_index]

        return eeg_signal, image_vector, text_vector