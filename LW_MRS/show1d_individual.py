import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import torch
import torch.fft as FFT
import argparse
import os
import time
import math

path = "C:/Users/s4548361/Desktop/Full/f2_proc.sec"
path2 = "C:/Users/s4548361/Desktop/Subsampled/f2_proc_sub.sec"
path3 = 'C:/Users/s4548361/Desktop/ML_test_data_easy_hsqc.tar/ML_test_data_easy_hsqc/Exp_Easy_HSQC/MaxEnt/maxent.sec'
path4 = 'C:/Users/s4548361/Desktop/ML_test_data_easy_hsqc.tar/ML_test_data_easy_hsqc/Exp_Easy_HSQC/FT_data/2DFT.sec'
path5 = 'C:/Users/s4548361/Desktop/ML_test_data_easy_hsqc.tar/ML_test_data_easy_hsqc/Exp_Easy_HSQC/FT_data/extrapolated/2DLPFT.sec'

path_test_full = 'C:/Users/s4548361/Desktop/MRS_new/test_data/full/f2_proc.sec'
path_test_sub = 'test_data/sub/f2_proc_sub.sec'
dicfull, datafull = ng.fileio.rnmrtk.read(
    path)

#
# Read in the .par/.sec files using NMR glue
dic, data = ng.fileio.rnmrtk.read(
     'result/LW5.sec')

def FFT1D(x):
    x = FFT.fft(x, dim = -1)
    x = FFT.fftshift(x, dim = -1)
    lenth = x.size()[-1]
    res = x / torch.sqrt(torch.tensor(lenth))
    return res

def AddNoise(ins, SNR):
    sigPower = SigPower(ins)
    noisePower = sigPower / SNR
    noise = torch.sqrt(noisePower.float()) * torch.randn(ins.size()).float()
    return ins + noise

def SigPower(ins):
    ll = torch.numel(ins)
    tmp1 = torch.sum(ins ** 2)
    return torch.div(tmp1, ll)

def DecayingTerm(ksp_coords, linewidth = 5, delta_T = 1 / 3648.8):
    ## ksp_coors: size: 1 * N; where N is the length of the spectrum;
    ## linewidth: size: Nb * 1, Nb is batch size;
    ## delta_T: can be a scalar or a vector of the same size as linewidth;
    LW_Kernel_img = np.exp(-1 * torch.pi * delta_T * linewidth * ksp_coords)
    return LW_Kernel_img

# dic, data = ng.fileio.rnmrtk.read(
#      path2)

udic = ng.fileio.rnmrtk.guess_udic(dic, data)

# Flip the data so that the last axis is the one we operate on
data = np.transpose(data)

# Unpack the STATES format into complex data in the last dimension
data = data[:, 0::2] + 1j * data[:, 1::2]

# data[:, 0] = data[:, 0] * 0.5

# # FFT the time domain axis
# data = np.fft.fftshift(np.fft.fft(data, axis=-1), -1)

label = data[585, :]
# scio.savemat('ista_global5.mat', {'real_part': np.real(label), 'imag_part': np.imag(label)})

## Normalization
abs_sub = np.absolute(label)

_range = max(abs_sub)

label = label / _range

Prob = torch.tensor(-2)
SNRs = torch.tensor([50, 40, 20, 10])

info = scio.loadmat('train_dataset_350_noise_free/train_data_1.mat')
mask = info['mask']

image = np.multiply(label, mask)

label = np.fft.fft(label, axis=-1)
label = np.fft.fftshift(label, axes=-1)
label = label / np.sqrt(350)

length = image.shape[-1]
ksp_coords = np.linspace(0, length - 1, length)
# linewidth = 5 + 5 * np.random.random()   ## random line-width, 5-10 Hz.
linewidth = 15
llw = linewidth
LW_kernels = DecayingTerm(ksp_coords, linewidth, delta_T=1 / 3648.8)
image = image * LW_kernels

linewidth = torch.tensor([linewidth])

image_r = np.real(image)
image_i = np.imag(image)
label_r = np.real(label)
label_i = np.imag(label)

## convert the image data to torch.tesors and return.
image_r = torch.from_numpy(image_r)
label_r = torch.from_numpy(label_r)
image_i = torch.from_numpy(image_i)
label_i = torch.from_numpy(label_i)

mask = torch.from_numpy(mask)

image_r = image_r.float()
label_r = label_r.float()
image_i = image_i.float()
label_i = label_i.float()
mask = mask.float()

### add noise into the input images;
tmp = torch.rand(1)
if tmp > Prob:
    # print('noise')
    # tmp_mask = lfs != 0
    # tmp_idx = torch.randint(4, (1, 1))
    tmp_idx = torch.tensor([0])
    tmp_SNR = SNRs[tmp_idx]
    image_r = AddNoise(image_r, tmp_SNR)
    image_r = image_r * mask
    image_i = AddNoise(image_i, tmp_SNR)
    image_i = image_i * mask

# plt.plot(np.absolute(label))
# plt.title('New_ISTA_LW5')
# plt.show()


image_mag = image_r + 1j * image_i
image_mag_spec = FFT1D(image_mag)
image_mag_spec_r = image_mag_spec.real
image_mag_spec_i = image_mag_spec.imag

test_imag_r = image_mag_spec_r.detach().numpy().astype(np.float32)
test_imag_i = image_mag_spec_i.detach().numpy().astype(np.float32)
scio.savemat('Label.mat', {'real_part': np.real(label), 'imag_part': np.imag(label)})
scio.savemat('Individual_lw'+str(llw) + '.mat', {'real_part': np.real(test_imag_r), 'imag_part': np.imag(test_imag_i)})

plt.plot(np.absolute(label))
plt.title('Label')
plt.show()

plt.plot(np.absolute(image_mag_spec.squeeze()))
plt.title('Individual_lw'+str(llw))
plt.show()





for iter_number in [12]:
    parser = argparse.ArgumentParser(description="MRS Demo")
    parser.add_argument("--cuda", action="store_true", help="use cuda?")
    parser.add_argument("--model", default="checkpoint_MRSNet_individual/model_epoch_100.pth", type=str, help="model path")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

    opt = parser.parse_args()
    cuda = opt.cuda


    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")




    model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

    # mask = np.loadtxt('test_format/1D_Subset_Coordinates/IN_SC_1.txt')[:, 1]
    # mask1 = mask
    # dic, k_data = ng.fileio.rnmrtk.read(
    #     'test_format/1D_Time_Full/IN_TF_C_1.sec')
    #
    # scale = np.abs(k_data[0])
    # k_data = k_data/scale
    # k_data[0] = 0.5 * k_data[0]
    # k_data[0] = 0.5 * k_data[0]


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.cuda()

    start_time = time.time()

    lw = linewidth

    image_r = torch.unsqueeze(image_r, 0)
    image_i = torch.unsqueeze(image_i, 0)
    mask = torch.unsqueeze(mask, 0)

    image_r = image_r.to(device)
    image_i = image_i.to(device)
    mask = mask.to(device)

    pred_r, pred_i, floss_r, floss_i = model(image_r, image_i, mask, lw, device=device)

    elapsed_time = time.time() - start_time

    # pred_r = iter_r[iter_number].cpu()
    # pred_i = iter_i[iter_number].cpu()

    pred_r = pred_r.cpu()
    pred_i = pred_i.cpu()

    pred_r = pred_r[0][0].detach().numpy().astype(np.float32)
    pred_i = pred_i[0][0].detach().numpy().astype(np.float32)

    spec_recon = pred_r + 1j * pred_i
    time_recon = math.sqrt(350) * np.fft.ifft(np.fft.ifftshift(spec_recon))
    time_recon = _range * time_recon

    spec_recon = np.fft.fftshift(np.fft.fft(time_recon)) / math.sqrt(350)
    time_recon[0] = 2 * time_recon[0]

    scio.savemat('Individual_recon' + str(llw) + '.mat',
                 {'real_part': np.real(spec_recon), 'imag_part': np.imag(spec_recon)})
    plt.plot(np.absolute(spec_recon))
    plt.title('Individual_recon'+str(llw))
    plt.show()

a= 1