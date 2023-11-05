__author__ = 'Silvia E Zieger'
__project__ = 'multi-analyte imaging using hyperspectral camera systems'

"""Copyright 2020. All rights reserved.

This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable 
for any damages arising from the use of this software.
Permission is granted to anyone to use this software within the scope of evaluating mutli-analyte sensing. No permission
is granted to use the software for commercial applications, and alter it or redistribute it.

This notice may not be removed or altered from any distribution.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pylab as plt
import matplotlib.patches as patches
from spectral import *
from scipy import ndimage
import seaborn as sns
import time
import pathlib
from glob import glob
import h5py
import os.path
import warnings
import xlrd

sns.set(style="ticks")
warnings.filterwarnings('ignore')


# =====================================================================================
def highlight_cell(x,y, width, height, ax=None, fill=False, **kwargs):
    rect = plt.Rectangle((x, y), width, height, fill=fill, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


# =====================================================================================
def load_cube(file_hdr, plot_cube=False):
    """
    Load the measurement file of the hyperspectral camera.
    :param file_hdr:    str specifying the location of the hdr file
    :param plot_cube:   boolean specifying whether the cubes shall be plotted or not; default False
    :return:
    """
    img_cube = open_image(file_hdr)
    integrationtime = float(img_cube.metadata['integration time'])
    wavelength = [float(l) for l in img_cube.metadata['wavelength']]

    parameter = {'cube': img_cube, 'Integration time': integrationtime, 'Wavelength': wavelength}

    # plotting cube for verification
    if plot_cube is True:
        img = img_cube.open_memmap()
        imshow(img)

    return parameter


def rotation_cube(img_cube, arg):
    """
    Rotate the hyperspectral cube in a certain angle (0, 90, 180, 270). Only quarter rotations are possible as of now.
    :param arg:     
    """
    if 'rotation' in arg:
        if arg['rotation'] == 0 or arg['rotation'] == 360:
            img = img_cube
        elif arg['rotation'] == 90:
            img_ = img_cube.swapaxes(0, 1)
            img = np.flip(img_, 0)
        elif arg['rotation'] == 180:
            img_ = np.flip(img_cube, 0)
            img = np.flip(img_, 1)
        elif arg['rotation'] == 270:
            img_ = np.flip(img_cube, 0)
            img = img_.swapaxes(0, 1)
        else:
            print('Rotation only possible in quarter anlges (n*90deg)')
            img = img_cube
    else:
        img = img_cube
    return img


def replace_NaN_with_Mean(df):
    for c in df.columns:
        ls_nan = np.where(np.asanyarray(np.isnan(df[c])))[0]
        for i in ls_nan:
            if c == 0:
                df.loc[i, c] = np.nanmean(df.loc[i-1:i+1, c:1])
            else:
                df.loc[i, c] = np.nanmean(df.loc[i-1:i+1, c-1:c+1]) 
    return df


def outlier_remove(df_check, blurr=False):
    q75, q25 = np.percentile(df_check, [75, 25])
    intr_qr = q75-q25
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    df_check = df_check.mask(df_check < min)
    df_check = df_check.mask(df_check > max)

    # if the nan vaues should be filled with the average of the surrounding values (aka blurr)
    if blurr is True:
        df_check = replace_NaN_with_Mean(df_check)
    return df_check


# =====================================================================================
def combine_rows4header(l1, l2, df):
    ls_head = list()
    for en, k in enumerate(l1):
        if 'Unnamed' in k:
            if isinstance(l2[en], float):
                ls_head.append('nan, nan')
            else:
                ls_head.append(' '.join([ls_head[en-1].split(' ')[0].strip(), str(l2[en])]))
        else:
            ls_head.append(' '.join([str(k), str(l2[en])]))

    # replace current header with created one
    df.columns = ls_head
    return df, ls_head


def boltzmann(x, top, bottom, V50, slope):
    """Boltzmann equation for pH; the parameter x equals the pH."""
    return bottom + (top - bottom) / (1 + 10**((x-V50)/slope)) 


def _sternvolmer_simple(x, f, k):
    """
    fitting function according to the common two site model. In general, x represents the pO2 or pCO2 content, whereas
    m, k and f are the common fitting parameters
    :param x:   list
    :param k:   np.float
    :param f:   np.float
    :return:
    """
    # tau0/tau
    tau_ratio = 1 / (f / (1. + k*x) + (1.-f))
    return tau_ratio
