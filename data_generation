import nmrglue as ng
import numpy as np
import os
import scipy.io as scio

train_num_samp = 1000
test_num_samp = 1024


train_path_256 = "C:/Users/s4548361/Desktop/MRS_old/train_dataset_256/"
train_path_512 = "C:/Users/s4548361/Desktop/MRS_old/train_dataset_512/"
test_path = "C:/Users/s4548361/Desktop/MRS_old/test_dataset_256/"
if not os.path.exists(train_path_256):
    os.makedirs(train_path_256)

if not os.path.exists(train_path_512):
    os.makedirs(train_path_512)

if not os.path.exists(test_path):
    os.makedirs(test_path)


for i in range(train_num_samp):

    dic, data = ng.fileio.rnmrtk.read(
        'C:/Users/s4548361/Desktop/random_256_64/1D_Time_Full/IN_TF_C_' + str(i+1) + '.sec')

    dic_sub, data_sub = ng.fileio.rnmrtk.read(
        'C:/Users/s4548361/Desktop/random_256_64/1D_Time_Subset/IN_TS_C_' + str(i+1) + '.sec')


    mask = data - data_sub

    data[0] = 0.5 * data[0]


    for a in range(len(mask)):
        if mask[a] == 0:
            mask[a] = 1
        else:
            mask[a] = 0
    mask = np.real(mask)


    os.chdir(train_path_256)
    scio.savemat('train_data_' + str(i+1) + '.mat', {'real': np.real(data), 'imag': np.imag(data), 'mask': mask})

for i in range(train_num_samp):

    dic, data = ng.fileio.rnmrtk.read(
        'C:/Users/s4548361/Desktop/random_512_64/1D_Time_Full/IN_TF_C_' + str(i+1) + '.sec')

    dic_sub, data_sub = ng.fileio.rnmrtk.read(
        'C:/Users/s4548361/Desktop/random_512_64/1D_Time_Subset/IN_TS_C_' + str(i+1) + '.sec')


    mask = data - data_sub

    data[0] = 0.5 * data[0]


    for a in range(len(mask)):
        if mask[a] == 0:
            mask[a] = 1
        else:
            mask[a] = 0
    mask = np.real(mask)


    os.chdir(train_path_512)
    scio.savemat('train_data_' + str(i+1) + '.mat', {'real': np.real(data), 'imag': np.imag(data), 'mask': mask})


for i in range(test_num_samp):

    dic, data = ng.fileio.rnmrtk.read(
        'R:/MLNMR2022-Q4755/For_Testing/2D_HSQC_1/input_time_full/1ds/' + str(i+1) + '.sec')

    dic_sub, data_sub = ng.fileio.rnmrtk.read(
        'R:/MLNMR2022-Q4755/For_Testing/2D_HSQC_1/input_time_sub/1ds/' + str(i+1) + '.sec')


    mask = data - data_sub

    data[0] = 0.5 * data[0]


    for a in range(len(mask)):
        if mask[a] == 0:
            mask[a] = 1
        else:
            mask[a] = 0
    mask = np.real(mask)


    os.chdir(test_path)
    scio.savemat('test_data_' + str(i+1) + '.mat', {'real': np.real(data), 'imag': np.imag(data), 'mask': mask})

print('Done')


