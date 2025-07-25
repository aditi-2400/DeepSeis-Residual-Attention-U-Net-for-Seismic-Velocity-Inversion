"""
dataset.py - Handles loading and preprocessing of seismic inversion dataset.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SeismicInversionDataset(Dataset):
    def __init__(self, sample_paths, downsample_factor=3):
        self.sample_paths = sample_paths
        self.downsample_factor = downsample_factor

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_path = self.sample_paths[idx]
        input_stack = []
        for i in [1, 75, 150, 225, 300]:
            file_path = os.path.join(sample_path, f"receiver_data_src_{i}.npy")
            data = np.load(file_path).astype(np.float32)
            data = data[::self.downsample_factor]
            input_stack.append(data)

        input_tensor = np.stack(input_stack, axis=0)  # shape: (5, T, 31)
        input_tensor = torch.from_numpy(input_tensor)

        target_path = os.path.join(sample_path, "vp_model.npy")
        target = np.load(target_path).astype(np.float32)
        target_tensor = torch.from_numpy(target).unsqueeze(0)
        return input_tensor, target_tensor
