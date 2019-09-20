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

dir_ = 'logs/sin1d3-1000'

frames = []
for i in range(0, 900):
    #im1 = readinimage(os.path.join(dir_, 'pred_{}.png'.format(i)))
    #im3 = readinimage(os.path.join(dir_, 'uw_{}_origin.png'.format(i)))    

    #im2 = readinimage(os.path.join(dir_, 'grad_{}.png'.format(i)))
    #print(im1.shape)
    #print(im2.shape)
    #im3 = readinimage(os.path.join(dir_, 'uw_origin_{}.png'.format(i)))    
    #print(i)
    #print(im3.shape)
    #im4 = readinimage(os.path.join(dir_, 'uw_resample_{}.png'.format(i)))
    #print(im4.shape)
    #im_up = np.concatenate([im1, im2], 1)
    #im_dn = np.concatenate([im3, im4], 1)

    im_up = readinimage(os.path.join(dir_, 't1t2n_origin_{}.png'.format(i)))
    im_dn = readinimage(os.path.join(dir_, 'ut2n_origin_{}.png'.format(i)))
    im = np.concatenate([im_up, im_dn], 0)
    #im = im[::2, ::2, :]
    frames.append(im)

imageio.mimwrite(os.path.join(dir_, 'show.mp4'), frames, fps=16)
