import torch
import os
from torch.utils.data import Dataset
import scipy.io as scio
import numpy as np
import math

class Dataset_512(Dataset):
    def __init__(self, root_path):
        super(Dataset, self).__init__()
        self.root_path = root_path
        files = os.listdir(root_path)
        self.components = []
        for file in files:
            path = self.root_path + file
            info = scio.loadmat(path)
            self.components.append({
                'full_r': info['real'],
                'full_i': info['imag'],
                'mask': info['mask']
            })

    def __getitem__(self, index):
        components = self.components[index]

        # For Self Generating data
        full_r = components['full_r']

        full_i = components['full_i']

        mask = components['mask']

        k_data = full_r + 1j * full_i
        k_samp_data = k_data * mask

        image_data = np.fft.fftshift(np.fft.fft(k_data))/math.sqrt(512)
        image_samp_data = np.fft.fftshift(np.fft.fft(k_samp_data))/math.sqrt(512)

        samp_r = np.real(k_samp_data)
        samp_i = np.imag(k_samp_data)
        samp_dft_r = np.real(image_samp_data)
        samp_dft_i = np.imag(image_samp_data)
        full_dft_r = np.real(image_data)
        full_dft_i = np.imag(image_data)

        samp_r = torch.from_numpy(samp_r).to(torch.float)
        samp_i = torch.from_numpy(samp_i).to(torch.float)
        samp_dft_r = torch.from_numpy(samp_dft_r).to(torch.float)
        samp_dft_i = torch.from_numpy(samp_dft_i).to(torch.float)
        full_dft_r = torch.from_numpy(full_dft_r).to(torch.float)
        full_dft_i = torch.from_numpy(full_dft_i).to(torch.float)
        mask = torch.from_numpy(mask).to(torch.float)

        return samp_r, samp_i, samp_dft_r, samp_dft_i, full_dft_r, full_dft_i, mask

    def __len__(self):
        return len(self.components)
