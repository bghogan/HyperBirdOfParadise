## Functions specific to the BOP data-set 

## 3D model angular information retrieval ##

from hsi.utils import GetFilename
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def GetUDLR(fp, bilplace, plot = False):
    """ Load the surface normal image for a given hyperspectral image, look in folder fp, using a .bil filename, optionally plot the images"""
    
    flr = fp + GetFilename(bilplace)+'_normal_R.tif'
    fud = fp + GetFilename(bilplace)+'_normal_G.tif'
    fma = fp + GetFilename(bilplace)+'_mask.png'
    fde = fp + GetFilename(bilplace)+'_depth.tif'

    ud = np.array(Image.open(fud))
    lr = np.array(Image.open(flr))
    ma = np.array(Image.open(fma))
    de = np.array(Image.open(fde))

    de = de - np.min(de)
    de = de / np.max(de)
    de[ma==0] = np.nan

    lr = lr/(2**16)
    lr = lr-0.5
    lr = lr * np.pi*2
    lr[ma==0] = np.nan

    ud = ud/(2**16)
    ud = ud-0.5
    ud = ud * np.pi*2
    ud[ma==0] = np.nan
    
    if plot == True:
        fig, ax = plt.subplots(1, 3, figsize = (10,10))

        i1 = ax[0].imshow(ud)
        fig.colorbar(i1, ax=ax[0])

        i2 = ax[1].imshow(lr)
        fig.colorbar(i2, ax=ax[1])

        i3 = ax[2].imshow(de)
        fig.colorbar(i3, ax=ax[2])

    return lr, ud, ma, de
    
    