import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt

import MRSNetConfig
import os
from pathlib import Path

MRSNetConfig.parser.description="MRS Show2D"
MRSNetConfig.readConfig('Global', 'Training', 'Demo')
MRSNetConfig.parser.add_argument('--basedir', type=Path, help='base directory for input and output data (leave blank for none)')
MRSNetConfig.parser.add_argument('--expdir', type=Path, help='[sub]directory for experiment data (default: blank)')
MRSNetConfig.parser.add_argument('--us_nmr_data', type=Path, help='[sub]directory for experiment data (default: blank)')
opt = MRSNetConfig.parser.parse_args()

path = "C:/Users/s4548361/Desktop/Full/f2_proc.sec"
path2 = "C:/Users/s4548361/Desktop/Subsampled/f2_proc_sub.sec"
path3 = 'C:/Users/s4548361/Desktop/ML_test_data_easy_hsqc.tar/ML_test_data_easy_hsqc/Exp_Easy_HSQC/MaxEnt/maxent.sec'
path4 = 'C:/Users/s4548361/Desktop/ML_test_data_easy_hsqc.tar/ML_test_data_easy_hsqc/Exp_Easy_HSQC/FT_data/2DFT.sec'
path5 = 'C:/Users/s4548361/Desktop/ML_test_data_easy_hsqc.tar/ML_test_data_easy_hsqc/Exp_Easy_HSQC/FT_data/extrapolated/2DLPFT.sec'

path_test_full = os.path.join(opt.basedir, opt.expdir, opt.us_nmr_data)

dicfull, datafull = ng.fileio.rnmrtk.read(
    path_test_full)

# Read in the .par/.sec files using NMR glue
dicmodel, datamodel = ng.fileio.rnmrtk.read(
     'result/global_LW5.sec')





def plotNMRData(dic,data,figpath=None):
    udic = ng.fileio.rnmrtk.guess_udic(dic, data)
    # Flip the data so that the last axis is the one we operate on
    data = np.transpose(data)

    # Unpack the STATES format into complex data in the last dimension
    data = data[:, 0::2] + 1j * data[:, 1::2]

    # FFT the time domain axis
    data = np.fft.fftshift(np.fft.fft(data, axis=-1), -1)

    # Repack the complex data back into STATES
    size = list(data.shape)
    half = int(size[-1])
    size[-1] = int(size[-1]) * 2
    d = np.empty(size, data.real.dtype)
    d[..., ::2] = data.real
    d[..., 1::2] = data.imag
    data = d

    # Flip the data back to the original format
    data = np.transpose(data)

    print(dic)
    print(udic)
    print(data.dtype, data.shape)

    # Create a contour plot so that we can view the data
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)

    # Create some contour levels for the plot
    lvl = np.max(data)
    scale = 1.4
    num_contours = 12
    pcl = [lvl * (1 / scale) ** x for x in range(num_contours)]
    list.reverse(pcl)
    colors = ['red'] * num_contours
    linewidths = [0.5] * (num_contours)

    # Use the udic to work out the axis scales
    ref_ppm_0 = udic[0]['car'] / udic[0]['obs']
    sw_ppm_0 = udic[0]['sw'] / udic[0]['obs']
    ref_ppm_1 = udic[1]['car'] / udic[1]['obs']
    sw_ppm_1 = udic[1]['sw'] / udic[1]['obs']

    y0 = ref_ppm_0 - sw_ppm_0 / 2
    y1 = ref_ppm_0 + sw_ppm_0 / 2 - sw_ppm_0 / (data.shape[0] / 2)

    x0 = ref_ppm_1
    x1 = ref_ppm_1 + sw_ppm_1 - sw_ppm_1 / data.shape[1]

    # Set the labels to display nice
    ax.set_ylim(y1, y0)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_xlim(x1, x0)
    extent = (x1, x0, y1, y0)

    ax.set_ylabel("15N ppm", size=20)
    ax.set_xlabel("1H ppm", size=20)
    ax.grid()
    ax.set_title('15N HSQC')

    # NOTE: I am using data[::2, :] which takes every second element from the data
    # matrix. This is gives me only the real data in the first dimension as it is STATES encoded
    ax.contour(data[::2, :], pcl, extent=extent,
	      colors=colors, linewidths=linewidths)
    plt.show()
    if figpath:
        plt.savefig(figpath)

#plotNMRData(dicfull,datafull,'show2d_full_no_noise.png')
plotNMRData(dicmodel,datamodel)
