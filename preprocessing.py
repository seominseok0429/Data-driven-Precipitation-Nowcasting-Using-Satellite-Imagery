import logging
import os
import pprint
import random
from tqdm import tqdm
import argparse

import numpy as np
import glob
import os

import netCDF4
import numpy
import glob

from scipy.interpolate import griddata
from scipy import ndimage

CT_path = './gk2a_conversion_table/'

def read_data(path):
    ncfile = netCDF4.Dataset(path,'r',format='netcdf4')
    ipixel=ncfile.variables['image_pixel_values']
    ipixel_process = ipixel[:]
    return ipixel_process, ipixel

def DQF_processing(data, ipixel):
    number_of_error_pixels = ipixel.getncattr('number_of_error_pixels')
    if (number_of_error_pixels > 0):
        data[data>49151] = 0
    return data

def pixel_masking(data, ipixel):
    channel=ipixel.getncattr('channel_name')
    if ((channel == 'VI004') or (channel == 'VI005') or (channel == 'NR016')):
        mask = 0b0000011111111111 #11bit mask
    elif ((channel == 'VI006') or (channel == 'NR013') or (channel == 'WV063')):
        mask = 0b0000111111111111 #12bit mask
    elif (channel == 'SW038'):
        mask = 0b0011111111111111 #14bit mask
    else:
        mask = 0b0001111111111111 #13bit mask
    ipixel_process_masked=numpy.bitwise_and(data, mask)
    return ipixel_process_masked, channel

def calibration(data, AL_postfix='_con_alb.txt', BT_postfix='_con_bt.txt', channel=None):
    if (channel[0:2] == 'VI') or (channel[0:2] == 'NR'):
        calibration_table=numpy.loadtxt(CT_path+channel+AL_postfix,'float64')
    else:
        calibration_table=numpy.loadtxt(CT_path+channel+BT_postfix,'float64')
    ipixel_process_masked_converted=calibration_table[data]
    return ipixel_process_masked_converted


def interpolate_nan_v1(data):
    data_interpolated = data.copy()
    nan_mask = np.isnan(data)
    data_interpolated[nan_mask] = ndimage.generic_filter(data, np.nanmean, size=3)[nan_mask]
    return data_interpolated

def interpolate_nan(data):
    """
    Interpolates NaN values in a 2D numpy array using griddata method.

    Parameters:
    data (numpy.ndarray): 2D array with NaN values to interpolate.

    Returns:
    numpy.ndarray: 2D array with NaN values interpolated.
    """
    # Create grid of coordinates
    x, y = np.indices(data.shape)

    # Mask valid points
    mask = ~np.isnan(data)

    # Interpolate only NaN values
    data_interpolated = griddata((x[mask], y[mask]), data[mask], (x, y), method='linear')

    return data_interpolated

ir_paths = glob.glob('/data/full_data/ir105/2021*/*/*/*.nc')
save_path = '/data/full_data/gk2a_npy/'

for i in tqdm(ir_paths):
    ir_path = i
    wv063_path = i.replace('ir105', 'wv063').replace('IR105', 'WV063')
    wv073_path = i.replace('ir105', 'wv073').replace('IR105', 'WV073')
    date = i.split('_')[-1].split('.')[0]

    ir105, ipixel = read_data(ir_path)
    ir105, channel = pixel_masking(ir105, ipixel)
    ir105 = calibration(ir105, channel=channel)
    ir105 = ir105[ 450 - 55: 450 + 245, 450 - 82:450 + 193]

    if np.isnan(ir105).any():
        ir105 = interpolate_nan(ir105)

    wv063, ipixel = read_data(wv063_path)
    wv063, channel = pixel_masking(wv063, ipixel)
    wv063 = calibration(wv063, channel=channel)
    wv063 = wv063[ 450 - 55: 450 + 245, 450 - 82:450 + 193]

    if np.isnan(wv063).any():
        wv063 = interpolate_nan(wv063)

    wv073, ipixel = read_data(wv073_path)
    wv073, channel = pixel_masking(wv073, ipixel)
    wv073 = calibration(wv073, channel=channel)
    wv073 = wv073[ 450 - 55: 450 + 245, 450 - 82:450 + 193]

    if np.isnan(wv073).any():
        wv073 = interpolate_nan(wv073)

    ir105 = (ir105 - 269.9956587688451) /15.605464610585223
    wv063 = (wv063 - 235.75187581685879) / 5.670746857074393
    wv073 = (wv073 - 250.51853481967774) / 8.350276602414484
    inputs = np.stack([ir105, wv063, wv073], axis=0)

    np.save(save_path + date + '.npy', inputs)

