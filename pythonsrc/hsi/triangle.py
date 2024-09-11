################ Functions for triangle space plots ################
## Utility functions for plotting triangles for 2d TCS plots.
# For example:
# MakeTriangle(axbig, rv, vlab, fcolor = 'RGB', ecolor = 'red', ann = 1)
#    sns.scatterplot(x = pltd[:,0], 
#                y = pltd[:,1], 
#               hue = [label_dict[species] for species in dat['meta']['species']], # relabel
#                palette = dict((label_dict[key], value) for (key, value) in color_dict.items()),
#                alpha=0.75, 
#                s=2,
#                edgecolor=None,
#                ax=axbig)

# More examples:
## We can also plot a 2d version of the tetrahedral colorspace
## array of vertex locations and labels
#v = np.array([[0, 0, .75], 
#      [-.61237, -.35355, -.25], 
#      [0, .70711, -.25], 
#      [.61237, -.35355, -.25]]) 
#vlab = np.array(['uv', 'b', 'g', 'r'])
#
## rotate xyz coordinates, and label coordinates to have 'uv' pointing toward us
#rv = triangle.RotateXYZtoTop(v, up = 'u')
#pltd = triangle.RotateXYZtoTop(txyz, up = 'u')
#
#fig, ax = plt.subplots(1)
#triangle.MakeTriangle(ax, rv, vlab, fcolor = 'RGB', ecolor = 'red', ann = 1)

## But we may not want uv to be up - here's some examples spinning the axes and data (signified by the text)
## you'd just want to apply RotateXYZtoTop to your tXYZ data too
#%matplotlib inline
#v = np.array([[0, 0, .75], 
#      [-.61237, -.35355, -.25], 
#      [0, .70711, -.25], 
#      [.61237, -.35355, -.25]]) 
#fig, ax = plt.subplots(1,4)
#vlab = np.array(['uv', 'b', 'g', 'r'])
#
#cc = np.array(['uv', 's', 'm', 'l'])
#for i in range(4):
#    rv = triangle.RotateXYZtoTop(v, up = cc[i])
#    triangle.MakeTriangle(ax[i], rv, vlab, fcolor = 'RGB', ecolor = 'red', ann = 1)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as patches
from scipy.spatial.transform import Rotation as R # used for rotating xyz according to which axis to 'look through' for plot
from matplotlib.colors import ListedColormap # used in MakeTriangle, to generate colormapping

def Dimage(sx, sy, c):
    """ Utility function makes diagonal gradient """
    c = [0, 0]
    img = np.ones([sx, sy])
    y = np.arange(0, img.shape[0], 1)
    x = np.arange(0, img.shape[1], 1)
    yy = (y - c[0])^2
    xx = (x - c[1])^2
    D = (xx[:,None] + yy[:,None].T) ** 2
    D = D / D.max()
    return D


def get_gradation_2d(start, stop, width, height, is_horizontal):
    """ Utility function makes linear gradient """
    if is_horizontal:
        D = np.tile(np.linspace(start, stop, width), (height, 1))
        D = D / D.max()
        return D
    else:
        D =  np.tile(np.linspace(start, stop, height), (width, 1)).T
        D = D / D.max()
        return D


def MakeRGBTriangle(sx = 512, sy = 512):
    """ Utility function to make RGB image gradients for triangle using 2 diagonal and one linear gradient
    sx and sy indicate the resolution of the interpolation used """
    
    c = [0,0] # center of gradient for diagonal
    # make diagonal gradient
    D = Dimage(sx, sy, c)
    # generate 2 diagonal and one linear gradient
    R = D
    L = np.rot90(D, k = 3)
    T = get_gradation_2d(sx, 0, sx, sy, 0)
    # stack to make color gradation rgb image
    RGB = np.stack((T, L, R), axis = 2)
    return RGB


def MakeAlphaColmap(Col):
    """ Utility function to make a colormap with defined hue to allow values other than plain RGB for triangle (we'll want a color for UV sometimes) """

    # Choose colormap
    cmap = plt.cm.RdBu
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))
    # Set alpha
    x = np.linspace(0, 1, cmap.N) ** 2
    x = x / x.max()
    # my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    my_cmap[:,-1] = x
    # Set color
    my_cmap[:,0:3] = Col
    # Create new colormap
    return(ListedColormap(my_cmap))
    # my_cmap()


def RotateXYZtoTop(v, up = 'u'):
    """ Rotate v (an sample by 3 numpy array) of x, y, z coordinates in tetrahedral color space to look down (~make parallel with z coordinate) whatever 
    axis is chosen. Note that 'u', 's', 'm' and 'l' correspond to rotations of the typical tetrahedral coordinates where stimulations (and or vertex positions) 
    are in usml order. 
    
    For example:

    v = np.array([[0, 0, .75], 
          [-.61237, -.35355, -.25], 
          [0, .70711, -.25], 
          [.61237, -.35355, -.25]]) 
    fig, ax = plt.subplots(1,4)
    vlab = np.array(['uv', 'b', 'g', 'r'])
    
    cc = np.array(['uv', 's', 'm', 'l'])
    for i in range(4):
        rv = RotateXYZtoTop(v, up = cc[i])
        MakeTriangle(ax[i], rv, vlab, fcolor = 'RGB', ecolor = 'red', ann = 1)
    
    """
    assert up in np.array(['u', 's', 'm', 'l']), "up must be either 'u', 's', 'm', or 'l'"

    if up == 'u':
        mult = 0 # rotating to begin with
        flip = 0 # flipping bottom most to topmost
        rot = 0 # rotate to get green up unless UV
    elif up == 'l':
        mult = 1
        flip = 110
        rot = 180+120
    elif up == 's':
        mult = 2
        flip = 110
        rot = 180-120
    elif up == 'm':
        mult = 3
        flip = 110
        rot = 180

    r = R.from_euler('zxz', [120*mult, flip, rot], degrees=True) # do rot on z to get the point you want at top at bottom, then flip up
    rv = r.apply(v)
    return rv


def MakeTriangle(ax, v = np.array([[0, 0, .75], 
          [-.61237, -.35355, -.25], 
          [0, .70711, -.25], 
          [.61237, -.35355, -.25]]), 
          vlab = np.array(['uv', 'b', 'g', 'r']), lab_colors = np.array([[1,0,1],[0,0,1],[0,1,0],[1,0,0]]), fcolor = 'None', ecolor = 'red', ann = 1):
    """ Function to plot out triangle with or without annotations, taking into account where vertex and labels colors should go.
    
    Arguments:
    ax - matplotlib plot axis label
    v - np array of vertex positions
    vlab - np array of vertex string labels
    lab_colors - np array of RGB colors for each vertex, if the face is colored - same order as the labels
    fcolor (facecolor) - either 'None' or 'RGB', either white or colored in background
    ecolor (edgecolor) - edge color for the triangle
    ann - integer 1 or 0 annotation (vertex labels) are on or off

    For example:

    v = np.array([[0, 0, .75], 
      [-.61237, -.35355, -.25], 
      [0, .70711, -.25], 
      [.61237, -.35355, -.25]]) # actual vertices, will want to label them, below is just for 3d plotting really
    rotate pltd and triangle vertices to get selected color to point up the z axis
    rv = RotateXYZtoTop(v, up = 'u')
    pltd = RotateXYZtoTop(pltd, up = 'u')
    MakeTriangle(ax, rv, vlab = np.array(['uv', 'b', 'g', 'r']), fcolor = bgcolor, ecolor = 'None', ann = 0)

    """
    
    ax.set_aspect('equal')
    
    # label each vertex
    if ann == 1:
        ax.text(v[0,0], v[0,1], vlab[0])
        ax.text(v[1,0], v[1,1], vlab[1])
        ax.text(v[2,0], v[2,1], vlab[2])
        ax.text(v[3,0], v[3,1], vlab[3])
    
    # color corners according to labels 
    c = lab_colors
    
    # Find and separate which vertice is within the others (pointing up), atm closest to zero in x and y axes
    lowest = np.argmin(np.sum(abs(v[:,(0,1)]), axis = 1))
    nverts = v[lowest, :]
    pverts = v[np.arange(len(v))!=lowest,:]
    c = c[np.arange(len(c))!=lowest,:]
    
    # reorder those pverts and colors to go top, left, right (order expected for plotting the colored triangle)
    i1 = np.argmax(pverts[:,1])
    i2 = np.argmin(pverts[:,0])
    i3 = np.argmax(pverts[:,0])
    pverts = pverts[(i1, i2, i3),:]
    tcolors = c[(i1, i2, i3),:]

    # determine size of box to zoom on for other plots
    xext = [pverts[:,0].min(), pverts[:,0].max()]
    yext = [pverts[:,1].min(), pverts[:,1].max()]
    
    if fcolor == 'RGB': # if want face/background colored
        
        # get gradients for applying colmap 
        RGB = MakeRGBTriangle()
        p = patches.Polygon(pverts[:,(0,1)], closed=True, facecolor='None', edgecolor=ecolor, zorder = 0)
        # add triangle patch to plot
        ax.add_patch(p)
        
        # for each corner, plot a colored point
        for i in range(3):
            im = ax.imshow(RGB[:,:,i], cmap = MakeAlphaColmap(tcolors[i]), extent = (xext[0], xext[1], yext[0], yext[1]), clip_on=True)
            im.set_clip_path(p)
        
    else:
    
        # generate plain triangle from pverts
        p = patches.Polygon(pverts[:,(0,1)], closed=True, facecolor=fcolor, edgecolor=ecolor, zorder = 0)
        ax.add_patch(p)
    
    # set xlims +- buffer size
    buff = 0.1
    ax.set_xlim([xext[0]-buff, xext[1]+buff])
    ax.set_ylim([yext[0]-buff, yext[1]+buff])
    