from __future__ import print_function

from glob import glob
import os

from fuel.datasets.hdf5 import H5PYDataset
import h5py
import pandas as pd
import numpy as np
from scipy.misc import imread


func = lambda x: int(x.split('-')[-1].split('.')[0])
seq_length = 50
img_dim = 256

f = h5py.File('/u/alitaiga/repositories/sim-to-real/robot_data.h5', mode='w')
# f = h5py.File('/data/lisatmp3/alitaiga/sim-to-real/robot_data.h5', mode='w')

cmds_rec1 = pd.read_csv('data/ergo-rec-1/data-cmds.csv')
pos_rec1 = pd.read_csv('data/ergo-rec-1/data-pos.csv')
vel_rec1 = pd.read_csv('data/ergo-rec-1/data-vel.csv')

cmds_rec2 = pd.read_csv('data/ergo-rec-2/data-cmds.csv')
pos_rec2 = pd.read_csv('data/ergo-rec-2/data-pos.csv')
vel_rec2 = pd.read_csv('data/ergo-rec-2/data-vel.csv')

func = lambda x: int(x.split('-')[-1].split('.')[0])

rec1_imgs = glob('data/ergo-rec-1/*.jpg')
rec1_imgs = sorted(liste, key=func)

rec2_imgs = glob('data/ergo-rec-2/*.jpg')
rec2_imgs = sorted(liste, key=func)

l1, l2 = [], []
cmds1, cmds2 = [], []
state1, state2 = [], []

for img1, img2 in zip(rec1_imgs, rec2_imgs):
    assert func(img1) == func(img2)
    im1, im2 = imread(img1), imread(img2)
    ind = func(img1)
    l1.append(im1.reshape(1,3,img_dim,img_dim))
    l2.append(im2.reshape(1,3,img_dim,img_dim))
    cmds1.append(cmds_rec1.iloc[ind].values)
    cmds2.append(cmds_rec2.iloc[ind].values)

    if len(l1) == seq_length:
        array1 = np.concatenate(l1, axis=0)
        array2 = np.concatenate(l2, axis=0)
