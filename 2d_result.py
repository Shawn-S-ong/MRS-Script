import argparse
import os
import torch
import time
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import torch.fft as FFT
import math
import torch.nn as nn
from DC_layer import DataConsistencyInKspace
from sklearn.metrics import mean_squared_error
import nmrglue as ng

parser = argparse.ArgumentParser(description="MRS Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="checkpoint_512/model_epoch_100.pth", type=str, help="model path")
# parser.add_argument("--model", default="checkpoint_256/model_epoch_100.pth", type=str, help="model path")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

opt = parser.parse_args()
cuda = opt.cuda
out_path = 'C:/Users/s4548361/Desktop/MRS_old/output_freq_AI_512/'

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

if not os.path.exists(out_path):
    os.makedirs(out_path)

## Fourier Transforms
def IFFT1D(x):
    return FFT.ifft(FFT.ifftshift(x), dim=-1, norm='ortho')


def FFT1D(x):
    return FFT.fftshift(FFT.fft(x, dim=-1, norm='ortho'))


model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]


test_data_len = 1024

data = np.zeros((512, 1024))


for i in range(test_data_len):


    info = scio.loadmat('test_dataset_256/test_data_' + str(i+1) + '.mat')

    # For Self Generating data

    full_r = info['real']
    full_i = info['imag']
    mask = info['mask']
    mask1 = mask
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

    mag_input = np.square(samp_dft_r) + np.square(samp_dft_i)


    samp_r = torch.from_numpy(samp_r).to(torch.float)
    samp_i = torch.from_numpy(samp_i).to(torch.float)
    samp_dft_r = torch.from_numpy(samp_dft_r).to(torch.float)
    samp_dft_i = torch.from_numpy(samp_dft_i).to(torch.float)
    full_dft_r = torch.from_numpy(full_dft_r).to(torch.float)
    full_dft_i = torch.from_numpy(full_dft_i).to(torch.float)
    mask = torch.from_numpy(mask).to(torch.float)

    samp_r = torch.unsqueeze(samp_r, 0)
    samp_i = torch.unsqueeze(samp_i, 0)
    samp_dft_r = torch.unsqueeze(samp_dft_r, 0)
    samp_dft_i = torch.unsqueeze(samp_dft_i, 0)
    mask = torch.unsqueeze(mask, 0)


    if cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    
    # Prediction Step
    ini_r, ini_i, y_r, y_i = model(samp_dft_r, samp_dft_i, samp_r, samp_i, mask)



    elapsed_time = time.time() - start_time

    ini_r = ini_r[0][0].detach().numpy().astype(np.float32)
    ini_i = ini_i[0][0].detach().numpy().astype(np.float32)
    y_r = y_r[0][0].detach().numpy().astype(np.float32)
    y_i = y_i[0][0].detach().numpy().astype(np.float32)




    dc_recon = y_r + 1j*y_i
    dc_recon_k = math.sqrt(256) * np.fft.ifft(np.fft.ifftshift(dc_recon))
    dc_recon_k[0] = 2 * dc_recon_k[0]

    dc_recon = np.fft.fftshift(np.fft.fft(dc_recon_k))/math.sqrt(256)
    new_data = []
    for count in range(len(dc_recon)):
        new_data.append(np.real(dc_recon)[count])
        new_data.append(np.imag(dc_recon)[count])

    data[:, i] = new_data


# Save the data in the required 2D format
dic, _ = ng.fileio.rnmrtk.read(
    'R:/MLNMR2022-Q4755/For_Testing/2D_HSQC_1/input_time_full/f2_proc.sec')
udic = ng.fileio.rnmrtk.guess_udic(dic, data)

dic['dom'] = ['F', 'F']
dic['npts'] = [256, 1024]
dic['nptype'] = ['C', 'R']
dic['layout'] = ([512, 1024], ['F1', 'F2'])
print(dic)

ng.fileio.rnmrtk.write(
    'new_data_512/fixed.sec', dic, data, overwrite=True)


print('Done')




