import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import imageio


def readinimage(path):
    I = Image.open(path)
    I = I.resize((608, 600))
    I_array = np.array(I)
    return I_array

dir_ = 'logs/cosc-1000-l12-2l-n3'

frames = []
for i in range(0, 600):
    #im1 = readinimage(os.path.join(dir_, 'pred_{}.png'.format(i)))
    #im3 = readinimage(os.path.join(dir_, 't1t2n_origin_{}.png'.format(i)))    
    im1 = readinimage(os.path.join(dir_, 'pred_{}.png'.format(i)))
    im2 = readinimage(os.path.join(dir_, 'f_0_samples_{}.png'.format(i)))
    #im3 = readinimage(os.path.join(dir_, 'f_1_samples_{}.png'.format(i)))
    #im4 = readinimage(os.path.join(dir_, 'f_2_samples_{}.png'.format(i)))
    #im5 = readinimage(os.path.join(dir_, 'f_3_samples_{}.png'.format(i)))
    #im2 = readinimage(os.path.join(dir_, 'theta2_samples_{}.png'.format(i)))
    #print(im1.shape)
    #print(im2.shape)
    #im3 = readinimage(os.path.join(dir_, 'uw_origin_{}.png'.format(i)))    
    #print(i)
    #print(im3.shape)
    #im4 = readinimage(os.path.join(dir_, 'ut2n_origin_{}.png'.format(i)))
    #print(im4.shape)
    #im_up = np.concatenate([im1, im2], 1)
    #im_dn = np.concatenate([im3, im4], 1)
    #im = np.concatenate([im_up, im_dn], 0)

    #im_up = np.concatenate([im1, im1], 1)
    #im_mi = np.concatenate([im2, im3], 1)
    #im_dn = np.concatenate([im4, im5], 1)
    #im = np.concatenate([im_up, im_mi, im_dn], 0)    
    im = np.concatenate([im1, im2], 1)
    #im_up = readinimage(os.path.join(dir_, 't1t2n_resample_{}.png'.format(i)))
    #im_dn = readinimage(os.path.join(dir_, 'ut2n_resample_{}.png'.format(i)))
    
    #im = np.concatenate([im1, im2, im3], 1)
    #im = im[::2, ::2, :]
    frames.append(im)

imageio.mimwrite(os.path.join(dir_, 'avideo.mp4'), frames, fps=16)
