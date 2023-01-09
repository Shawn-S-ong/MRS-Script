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

from sklearn.metrics import mean_squared_error
import nmrglue as ng

for iter_number in [12]:
    parser = argparse.ArgumentParser(description="MRS Demo")
    parser.add_argument("--cuda", action="store_true", help="use cuda?")
    parser.add_argument("--model", default="checkpoint_MRSNet_global/model_epoch_100.pth", type=str, help="model path")
    # parser.add_argument("--model", default="checkpoint_256/model_epoch_100.pth", type=str, help="model path")
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

    test_data_len = 1024
    test_num_samp = 1024

    recon_data = np.zeros((700, 1024))
    spec_2d_recon = np.zeros((700, 1024))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    path_test_full = "C:/Users/s4548361/Desktop/Exp_Easy_HSQC/Raw_data/Full/f2_proc.sec"
    path_test_sub = "C:/Users/s4548361/Desktop/Exp_Easy_HSQC/Raw_data/Subsampled/f2_proc_sub.sec"


    _, sub_data = ng.fileio.rnmrtk.read(path_test_sub)
    _, full_data = ng.fileio.rnmrtk.read(path_test_full)

    sub_data = np.transpose(sub_data)
    sub_data = sub_data[:, 0::2] + 1j * sub_data[:, 1::2]

    full_data = np.transpose(full_data)
    full_data = full_data[:, 0::2] + 1j * full_data[:, 1::2]


    for i in range(test_num_samp):

        data = full_data[i]

        data_sub = sub_data[i]

        mask = data - data_sub

        data_sub[0] = 0.5 * data_sub[0]

        data[0] = 0.5 * data[0]
        ## Normalization
        abs_sub = np.absolute(data_sub)

        _range = max(abs_sub)

        data = data / _range

        for a in range(len(mask)):
            if mask[a] == 0:
                mask[a] = 1
            else:
                mask[a] = 0
        mask = np.real(mask)

        full_r = np.real(data)
        full_i = np.imag(data)

        mask1 = mask
        label = full_r + 1j * full_i
        image = label * mask

        label = np.fft.fftshift(np.fft.fft(label)) / math.sqrt(350)
        input = np.fft.fftshift(np.fft.fft(image)) / math.sqrt(350)

        image_r = np.real(image)
        image_i = np.imag(image)
        label_r = np.real(label)
        label_i = np.imag(label)
        input_r = np.real(input)
        input_i = np.imag(input)

        image_r = torch.from_numpy(image_r).to(torch.float)
        image_i = torch.from_numpy(image_i).to(torch.float)
        label_r = torch.from_numpy(label_r).to(torch.float)
        label_i = torch.from_numpy(label_i).to(torch.float)
        mask = torch.from_numpy(mask).to(torch.float)

        image_r = image_r.float()
        label_r = label_r.float()
        image_i = image_i.float()
        label_i = label_i.float()
        mask = mask.float()

        image_r = torch.unsqueeze(image_r, 0)
        label_r = torch.unsqueeze(label_r, 0)
        image_i = torch.unsqueeze(image_i, 0)
        label_i = torch.unsqueeze(label_i, 0)
        mask = torch.unsqueeze(mask, 0)

        image_r = torch.unsqueeze(image_r, 0)
        label_r = torch.unsqueeze(label_r, 0)
        image_i = torch.unsqueeze(image_i, 0)
        label_i = torch.unsqueeze(label_i, 0)
        mask = torch.unsqueeze(mask, 0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = model.cuda()

        start_time = time.time()

        lw = torch.tensor([5])

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


        mid_storage = []
        for count in range(len(spec_recon)):
            mid_storage.append(np.real(spec_recon)[count])
            mid_storage.append(np.imag(spec_recon)[count])

        spec_2d_recon[:, i] = mid_storage

        new_data = []
        for count in range(len(time_recon)):
            new_data.append(np.real(time_recon)[count])
            new_data.append(np.imag(time_recon)[count])

        recon_data[:, i] = new_data
        print(str(i))


    # scio.savemat('final.mat', {'recon': spec_2d_recon})


    dic, _ = ng.fileio.rnmrtk.read(
        "C:/Users/s4548361/Desktop/Exp_Easy_HSQC/Raw_data/Full/f2_proc.sec")
    udic = ng.fileio.rnmrtk.guess_udic(dic, recon_data)

    dic['dom'] = ['F', 'F']
    dic['npts'] = [350, 1024]
    dic['nptype'] = ['C', 'R']
    dic['layout'] = ([700, 1024], ['F1', 'F2'])
    print(dic)

    ng.fileio.rnmrtk.write(
        'result/global_LW5' + '.sec', dic, recon_data, overwrite=True)


    print('Done')




