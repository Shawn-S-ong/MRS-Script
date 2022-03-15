import torch
import os
from torch.utils.data import Dataset
import scipy.io as scio
import numpy as np
import math

class Dataset(Dataset):
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

        image_data = np.fft.fftshift(np.fft.fft(k_data))/math.sqrt(256)
        image_samp_data = np.fft.fftshift(np.fft.fft(k_samp_data))/math.sqrt(256)

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

        # samp_r = torch.unsqueeze(samp_r, 0)
        # samp_i = torch.unsqueeze(samp_i, 0)
        # samp_dft_r = torch.unsqueeze(samp_dft_r, 0)
        # samp_dft_i = torch.unsqueeze(samp_dft_i, 0)
        # full_dft_r = torch.unsqueeze(full_dft_r, 0)
        # full_dft_i = torch.unsqueeze(full_dft_i, 0)
        # mask = torch.unsqueeze(mask, 0)

        ## For Real data

        # # components['samp_r'][0] = 0.5 * components['samp_r'][0]
        # samp_r = torch.from_numpy(components['samp_r']).to(torch.float)
        # samp_r = torch.unsqueeze(samp_r, 0)
        # # components['samp_i'][0] = 0.5 * components['samp_i'][0]
        # samp_i = torch.from_numpy(components['samp_i']).to(torch.float)
        # samp_i = torch.unsqueeze(samp_i, 0)
        # # full_r = torch.from_numpy(components['full_r']).to(torch.float)
        # # full_r = torch.unsqueeze(full_r, 0)
        # # full_i = torch.from_numpy(components['full_i']).to(torch.float)
        # # full_i = torch.unsqueeze(full_i, 0)
        # samp_dft_r = torch.from_numpy(components['samp_dft_r']).to(torch.float)
        # samp_dft_r = torch.unsqueeze(samp_dft_r, 0)
        # samp_dft_i = torch.from_numpy(components['samp_dft_i']).to(torch.float)
        # samp_dft_i = torch.unsqueeze(samp_dft_i, 0)
        # full_dft_r = torch.from_numpy(components['full_dft_r']).to(torch.float)
        # full_dft_r = torch.unsqueeze(full_dft_r, 0)
        # full_dft_i = torch.from_numpy(components['full_dft_i']).to(torch.float)
        # full_dft_i = torch.unsqueeze(full_dft_i, 0)
        # mask = torch.from_numpy(components['mask']).to(torch.float)
        # mask = torch.unsqueeze(mask, 0)

        return samp_r, samp_i, samp_dft_r, samp_dft_i, full_dft_r, full_dft_i, mask

    def __len__(self):
        return len(self.components)
