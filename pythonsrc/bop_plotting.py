## Functions specific to the BOP data-set 

## PLOTTING ##
# Note that virtually all plots depend on having pickled dataset already generated (i.e. bop_analysis.RunDataAnalysis() has been run)

from scipy import io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
import colour
import spectral.io.envi as envi 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import patches
from matplotlib.colors import Normalize
from scipy.ndimage import binary_dilation
from scipy.ndimage import generate_binary_structure
import re

# hsi module (generic utilities)
import hsi.tetra as tetra
import hsi.utils as utils
import hsi.triangle as triangle

# used in 3d plots
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default = "browser"

# used to load analyzed data, and some kde plots
import bop_analysis
import seaborn as sns
from scipy.stats import median_abs_deviation
from scipy.stats import gaussian_kde

def BasicPlot(pltn, mat_contents, label_dict, color_dict, lab = None, 
              maskin = [0, 2, 3, 4, 6], reembed = False, n_neighbors = 15, min_dist = 0.1, sort_by_rgb = False):

    """ This plots a 5*2 or 3 panel of the first 2 dimensions of 'pltn', columns are patches, first row is sRGB, 2nd row kde plotted by species, and 
    optional a third row showing selected spectra (corresponding to points near the highest density kde of pltn for each species). Third row
    is added if pltn in: 'dSh_umap_embedded', 'KLPD_umap_embedded', 'pca_umap_embedded', 'pca_umap_embedded_bright' """
    
    bgcolor = '#5A5A5A'
    hasSpec = ['dSh_umap_embedded', 'KLPD_umap_embedded', 'pca_umap_embedded', 'pca_umap_embedded_bright']

    # if we want a row of panels containing spectra for points of high density from the KDEs, add a row
    if pltn in hasSpec:
        fig, ax = plt.subplots(nrows=3, ncols=len(maskin), figsize=(2.42*len(maskin), 2.42*3))
    else:
        fig, ax = plt.subplots(nrows=2, ncols=len(maskin), figsize=(2.42*len(maskin), 2.42*2))
    fig.tight_layout()
   
    # for each mask, generate 2 or 3 panels in a column
    for i in range(len(maskin)):
        print('Working on ' + str(i+1) + ' of ' + str(len(maskin)))

        # load the relevant data
        if maskin[i]==2: # for breast, use tilted image samples instead
            dat = bop_analysis.LoadGather(maskin[i], directory = '../dat/Tilted_hyperspectral_images')
        else:
            dat = bop_analysis.LoadGather(maskin[i], directory = '../dat/Normal_hyperspectral_images')
            
        # since we're plotting some colors, can re-generate RGB, this allows us to dynamically
        # update color scaling used for plotting if wanted.
        # XYZ, sRGB, Lab = utils.GetColors(dat['arr'], dat['wl'])
        sRGB = dat['sRGB']

        # Choose relevant distances for plotting
        if reembed:
            # Using reembed = True allows us to make plots with altered UMAP parameters, but it's only defined for pcasam currently
            pltd = utils.EmbedDist(Coordinates = dat['pcasam_out'], n_neighbors = n_neighbors, min_dist = min_dist) # calculate embeddings again
            print('Plotting umap embed of PCA (with subtraction of mean), re-embedding with provided n_neighbors and min dist')
        else:
            pltd = dat[pltn]

        # If we're plotting LAB space, and the 'lab' argument is defined, reorder pltd temporarily to set lva, avb, or lvb
        if lab == 'lva' and pltn == 'Lab':
            pltd = pltd[:,(0,1)]
            print('Plotting lva')
        elif lab == 'avb' and pltn == 'Lab':
            pltd = pltd[:,(1,2)]
            print('Plotting avb')
        elif lab == 'lvb' and pltn == 'Lab':
            pltd = pltd[:,(0,2)]
            print('Plotting lvb')
        
        # If we're looking at PCA with brightness still there, you can pick here if you want to omit the 1st pca axis, by default we do
        if pltn == 'pcasam_out_bright':
            pltd = pltd[:,(1,2)] # omit 1st brighness axis
            # pltd = pltd[:,(0,1)] # or dont

        # Set title to the patch name
        tit = mat_contents.bwlist[maskin[i]].replace('Bottom', '')
        ax[0,i].title.set_text(tit.replace('Belly', 'Belly/vent'))
        
        # If we want to control order the plotting by the brightness of the points so dark points don't hide bright ones
        if sort_by_rgb == True:
            sortorder = np.argsort(np.sum(sRGB, axis = 1)) 
        else:
            sortorder = np.arange(pltd.shape[0]) # retain order of original sampling
            np.random.shuffle(sortorder) # then randomize the order

        # sRGB scatter plot for first row of panels
        ax[0,i].scatter(x = pltd[sortorder,0], 
                        y = pltd[sortorder,1], 
                        c = sRGB[sortorder,:], 
                        alpha=.8, edgecolors=None)
        ax[0,i].set_facecolor(bgcolor)
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
        
        # get plot limits
        left, right = ax[0,i].get_xlim()
        bottom, top = ax[0,i].get_ylim()

        # kde plot for second row
        sns.kdeplot(x = pltd[:,0], 
                y = pltd[:,1], 
                hue = [label_dict[species] for species in dat['meta']['species']], # relabel colors with dict
                palette = dict((label_dict[key], value) for (key, value) in color_dict.items()), # relabel colors with dict
                fill=False,
                thresh=0, 
                common_norm = False, 
                legend = False,
                levels=[0.2], # [0.2, 0.6, 0.8],
                ax=ax[1,i])
        ax[1,i].set_facecolor(bgcolor)
        ax[1,i].set_xlim(left,right)
        ax[1,i].set_ylim(bottom, top)
        ax[1,i].set_xticks([])
        ax[1,i].set_yticks([])
        
        # if this plotting variable is noted as being appropriate for adding a 3rd row, plotting spectra
        if pltn in hasSpec:

            if i==0: # no point in plotting spectra for the 'whole' patch
                ax[2,i].axis('off')

            if i>0: # for any other patch

                # list the species
                listsp = np.asarray([label_dict[species] for species in dat['meta']['species']])
                thing = np.unique(listsp)
                yl = 0

                for x, zz in zip(thing, range(len(thing))): # for each species

                    # get x and y embedded positions for species
                    kx, ky = pltd[listsp==x,0], pltd[listsp==x,1]

                    # use plot lims for kernel
                    xmin, xmax = np.min(pltd[:,0]), np.max(pltd[:,0])
                    ymin, ymax = np.min(pltd[:,1]), np.max(pltd[:,1])

                    # generate kernel to find area of max density
                    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    # apply kernel to x and y from species
                    kernel = gaussian_kde(np.vstack([kx, ky]))
                    Z = np.reshape(kernel(positions).T, X.shape)
                    # predict with kernel all values including other species 
                    # (to get back to original array shape)
                    val_pdf = kernel.pdf(pltd.T)
                    val_pdf[listsp!=x]=float('-inf')

                    # find max value
                    max_point = np.argmax(val_pdf)

                    # find distance of all embedded points to our selected point of maximum density
                    n_spec = 50
                    curr_point_dist = np.linalg.norm(np.subtract(pltd, pltd[max_point]), axis = 1)

                    # record the closest n idx's and closest location
                    closest_point_idx = curr_point_dist.argsort()[:n_spec]
                    closest_point_xy = pltd[curr_point_dist.argsort()[:1],:]

                    # get the plotting vars then
                    y_plot = dat['arr'][closest_point_idx,:].T
                    x_plot = dat['wl']

                    # get median and mad for those points
                    me_plot = np.median(y_plot, axis = 1)
                    sd_plot = median_abs_deviation(y_plot, axis = 1)

                    # get species color
                    scol = dict((label_dict[key], value) for (key, value) in color_dict.items()).get(x)

                    # scatter the points of max density, colored by species, on the kde plot
                    ax[1,i].scatter(pltd[max_point,0], pltd[max_point,1], s = 50, color = scol, 
                                    edgecolors = 'black', linewidths = 1)

                    wl_mask = x_plot<325

                    # plot median
                    ln = ax[2,i].plot(x_plot[~wl_mask], me_plot[~wl_mask], color = scol, linewidth = 2.5, label = x)

                    # get mean colors too
                    plt_XYZ, plt_sRGB, plt_Lab = utils.GetColors(me_plot, x_plot)

                    # plot mad
                    ax[2,i].fill_between(x_plot[~wl_mask], me_plot[~wl_mask]-sd_plot[~wl_mask], me_plot[~wl_mask]+sd_plot[~wl_mask], color = plt_sRGB, alpha = 0.75)
                    ax[2,i].plot(x_plot[wl_mask], me_plot[wl_mask], color = scol, linewidth = 1.5, linestyle = 'dotted')
                    ax[2,i].set_xticks(np.arange(300, 701, 100))
                    
                    # find y limit for plot, keep max of any iterations here
                    top = max(me_plot[~wl_mask]+sd_plot[~wl_mask]) * 1.2
                    yl = max(yl, top)                
                
                ax[2,i].set_ylim([0, yl])
                # ax[2,i].set_ylim([0,0.6])
                ax[2,i].set_facecolor(bgcolor)

    plt.show()
    return fig
    
def CIEXYyPlot_two(mat_contents, includeWhole, label_dict, color_dict, sort_by_rgb = True):

    """ Plots one large CIEXyy plot, and next to this adds 5*2 subplots where columns are patch, 
    first row is sRGB points scattered on a CIEXyy space, and 2nd row is same points with kde by species. 
    Note: some blank panels are plotted during generation """

    print('Expect CIEXYyPlot_two to generate some blank plots in-line, but still produces correct saved output')

    maskin = [0, 2, 3, 4, 6] # only main ones patches
    bgcolor = '#5A5A5A'

    # Should we have all mask columns, or skip first (whole)?
    if includeWhole == 1:
        fig, ax = plt.subplots(ncols=len(maskin)+1, nrows=2, figsize=(2.42*(len(maskin)+1), 2.42*2))
        offset = 1
    else:
        fig, ax = plt.subplots(ncols=len(maskin), nrows=2, figsize=(2.42*len(maskin), 2.42*2))
        offset = 0

    # Make first column contain all rows
    gs = ax[1, 2].get_gridspec()
    for a in ax[0:, 0]:
        a.remove()
    axbig = fig.add_subplot(gs[0:, 0])    
    fig.tight_layout()

    # for each mask
    for i in range(len(maskin)): 
        print('Working on ' + str(i+1) + ' of ' + str(len(maskin)))

        # load the relevant pickled data
        if maskin[i]==2: # for breast, use tilted image samples instead of 'normal' ones
            dat = bop_analysis.LoadGather(maskin[i], directory = '../dat/Tilted_hyperspectral_images/')
        else:
            dat = bop_analysis.LoadGather(maskin[i], directory = '../dat/Normal_hyperspectral_images/')
            
        # since we're plotting some colors, we could re-generate RGB, this allows us to dynamically
        # update color scaling used for plotting if wanted.
        # XYZ, sRGB, Lab = utils.GetColors(dat['arr'], dat['wl'])
        sRGB = dat['sRGB']
        
        # choose relevant distances etc
        pltn = 'XYZ'
        pltd = colour.XYZ_to_xy(dat[pltn]) # convert CIE XYZ tristimulus values into CIE xyY space

        if sort_by_rgb == True:
            sortorder = np.argsort(np.sum(sRGB, axis = 1)) # order the plotting by the brightness of the points so dark points don't hide bright ones
        else:
            sortorder = np.arange(pltd.shape[0]) # retain order of original sampling
            np.random.shuffle(sortorder) # then randomize the order

        # plot for big axis
        if i==0:

            xl=[0,0.75]
            yl=[0,0.85]

            sns.scatterplot(x = pltd[:,0], 
                    y = pltd[:,1], 
                    hue = [label_dict[species] for species in dat['meta']['species']], # relabel
                    palette = dict((label_dict[key], value) for (key, value) in color_dict.items()),    
                    alpha=0.75, 
                    s=2,
                    edgecolor=None,
                    legend = False,
                    ax=axbig)

            fig, _ = colour.plotting.diagrams.plot_chromaticity_diagram(show=False, 
                                                          title = "Test 1931 CIE plot",
                                                          show_diagram_colours=True,
                                                          figure = fig, axes = axbig,
                                                          tight_layout = False)

            
            tit = mat_contents.bwlist[maskin[i]].replace('Bottom', '')
            axbig.title.set_text(tit.replace('Belly', 'Belly/vent'))
            axbig.set_xlim(xl)
            axbig.set_ylim(yl)

            xl=[min(pltd[:,0]),max(pltd[:,0])]
            yl=[min(pltd[:,1]),max(pltd[:,1])]

            rect=patches.Rectangle([xl[0], yl[0]], xl[1]-xl[0], yl[1]-yl[0], fill = 0, color = 'r')
            axbig.add_patch(rect)

        if includeWhole == 1 or i>0:

            # sRGB plot for first row
            ax[0,i+offset].scatter(x = pltd[sortorder,0], 
                            y = pltd[sortorder,1], 
                            c = sRGB[sortorder,:], 
                            alpha=1, edgecolors=None)
            tit = mat_contents.bwlist[maskin[i]].replace('Bottom', '')
            ax[0,i+offset].title.set_text(tit.replace('Belly', 'Belly/vent'))
            
            ## kde plot for second row
            sns.kdeplot(x = pltd[:,0], 
                    y = pltd[:,1], 
                    hue = [label_dict[species] for species in dat['meta']['species']], # relabel
                    palette = dict((label_dict[key], value) for (key, value) in color_dict.items()),
                    fill=False, thresh=0, 
                    common_norm = False,legend=False,
                    levels=[0.2], # [0.2, 0.6, 0.8],
                    ax=ax[1,i+offset])
            
            # add chromaticity diagram to the scatter plot.
            fig, ax[0,i+offset] = colour.plotting.diagrams.plot_chromaticity_diagram(show=False, 
                           title = False,
                          show_diagram_colours=False,
                          figure = fig, axes = ax[0,i+offset],
                          show_spectral_locus = True,
                          tight_layout = False)
            
            # and to the density plot
            fig, ax[1,i+offset] = colour.plotting.diagrams.plot_chromaticity_diagram(show=False, 
                                                   title = False,
                                                  show_diagram_colours=False,
                                                  figure = fig, axes = ax[1,i+offset],
                                                  show_spectral_locus = True,
                                                  tight_layout = False)

            ax[1,i+offset].set_xlim(xl)
            ax[1,i+offset].set_ylim(yl)
            ax[0,i+offset].set_xlim(xl)
            ax[0,i+offset].set_ylim(yl)
            
    return fig
            

## 2d Triangular plot of tetrahedral XYZ,  allowing us to choose which axis is pointing up (toward us)
def tXYZplot(mat_contents, includeWhole,  label_dict, color_dict, up = 'u', sort_by_rgb = True):

    """ 2D plot of tetrahedral color space, columns patch and top row sRGB scatter, 2nd row kde by species. """

    maskin = [0, 2, 3, 4, 6] # only main patches
    bgcolor = '#5A5A5A'
    
    # do we want first mask 'whole' in expanded panel?
    if includeWhole == 1:
        widths = np.repeat([1], [len(maskin)+1]).tolist()
        widths[0]=2
        fig, ax = plt.subplots(ncols=len(maskin)+1, nrows=2, figsize=(2.42*(len(maskin)+1), 2.42*2),
                               gridspec_kw={'width_ratios': widths})
        offset = 1
    else:
        widths = np.repeat([0], [len(maskin)]).tolist()
        widths[0]=1.01
        fig, ax = plt.subplots(ncols=len(maskin), nrows=2, figsize=(2.42*len(maskin), 2.42*2), 
                               gridspec_kw={'width_ratios': widths})
        offset = 0
        
    # remove the underlying axes, make big axis
    gs = ax[1, 2].get_gridspec()
    for a in ax[0:, 0]:
        a.remove()
    axbig = fig.add_subplot(gs[0:, 0])
    fig.tight_layout()
        
    # what do we want to point upward, and labels
    vlab = np.array(['uv', 'b', 'g', 'r'])
    clab = np.array([[1,0,1],[0,0,1],[0,1,0],[1,0,0]])
    v = np.array([[0, 0, .75], 
          [-.61237, -.35355, -.25], 
          [0, .70711, -.25], 
          [.61237, -.35355, -.25]]) # actual vertices, will want to label them, below is just for 3d plotting really

    for i in range(len(maskin)):
        
        print('Working on ' + str(i+1) + ' of ' + str(len(maskin)))

        # load the relevant data
        if maskin[i]==2: # for breast, use tilted image samples instead
            dat = bop_analysis.LoadGather(maskin[i], directory = '../dat/Tilted_hyperspectral_images/')
        else:
            dat = bop_analysis.LoadGather(maskin[i], directory = '../dat/Normal_hyperspectral_images/')
            
        # choose relevant distances etc
        pltn = 'txyz'
        pltd =  dat[pltn]
            
        # since we're plotting some colors, can re-generate RGB, this allows us to dynamically
        # update color scaling used for plotting if wanted.
        # XYZ, sRGB, Lab = utils.GetColors(dat['arr'], dat['wl'])
        sRGB = dat['sRGB']
        if sort_by_rgb == True:
            sortorder = np.argsort(np.sum(sRGB, axis = 1)) # order the plotting by the brightness of the points so dark points don't hide bright ones
        else:
            sortorder = np.arange(pltd.shape[0]) # retain order of original sampling
            np.random.shuffle(sortorder) # then randomize the order
        
        # rotate pltd and triangle vertices to get selected color to point up the z axis
        rv = triangle.RotateXYZtoTop(v, up = up)
        pltd = triangle.RotateXYZtoTop(pltd, up = up)

        # plot for big axis
        if i==0:

            # plot the triangle, and color it
            triangle.MakeTriangle(axbig, v = rv, vlab = vlab, lab_colors = clab, fcolor = 'RGB', ecolor = 'red', ann = 0)

            # scatter all samples, color accoring to species
            sns.scatterplot(x = pltd[:,0], 
                        y = pltd[:,1], 
                        hue = [label_dict[species] for species in dat['meta']['species']], # relabel
                        palette = dict((label_dict[key], value) for (key, value) in color_dict.items()),
                        alpha=0.75, 
                        s=2,
                        edgecolor=None,
                        ax=axbig,
                           legend = False) # Should mix these up in order - now H overplots the others

            # determine box to zoom on for other patches
            xl=[min(pltd[:,0]),max(pltd[:,0])]
            yl=[min(pltd[:,1]),max(pltd[:,1])]

            # draw selected box
            rect=patches.Rectangle([xl[0], yl[0]], xl[1]-xl[0], yl[1]-yl[0], fill = 0, color = 'black')
            axbig.add_patch(rect)
            
            # hide axes
            axbig.axis('off')

        if includeWhole == 1 or i>0:

            # sRGB plot for third row
            triangle.MakeTriangle(ax[0,i+offset], v = rv, vlab = vlab, lab_colors = clab, fcolor = bgcolor, ecolor = 'None', ann = 0)

            ax[0,i+offset].scatter(x = pltd[sortorder,0], 
                            y = pltd[sortorder,1], 
                            c = sRGB[sortorder,:], 
                            alpha=1, edgecolors=None)
            ax[0,i+offset].set_xlim(xl)
            ax[0,i+offset].set_ylim(yl)
            
            tit = mat_contents.bwlist[maskin[i]].replace('Bottom', '')
            ax[0,i+offset].title.set_text(tit.replace('Belly', 'Belly/vent'))

            # kde plot for second row
            triangle.MakeTriangle(ax[1,i+offset], v = rv, vlab = vlab, lab_colors = clab, fcolor = bgcolor, ecolor = 'None', ann = 0)

            sns.kdeplot(x = pltd[:,0], 
                    y = pltd[:,1], 
                    hue = [label_dict[species] for species in dat['meta']['species']], # relabel
                    palette = dict((label_dict[key], value) for (key, value) in color_dict.items()),
                    fill=False, thresh=0, 
                    common_norm = False,legend=False,
                    levels=[0.2], # [0.2, 0.6, 0.8],
                    ax=ax[1,i+offset])
            ax[1,i+offset].set_xlim(xl)
            ax[1,i+offset].set_ylim(yl)
            # ax[1,i+offset].set_aspect('auto')

    return fig
        
def SampleCheck_half(pas = range(0,7), fname = '', halfSpecimens = True, lookin = False, color_dict = dict({'CM_M':[230/255,159/255,0/255],
                  'CR_M':[0/255,158/255,115/255],
                  'H_M': [86/255,180/255,233/255]}), saveplot = True, scale_factor = 4):

    """ Plots an outline of the patch mask, and all samples collected for that image/patch combination, for all images by patch. This is a very memory intensive function. """
    
    # dilation structure
    nrep = 5 # in combo with iterations = 5, boundary of 25 pixels
    bin_struc = np.repeat(np.repeat(generate_binary_structure(2, 1), nrep, axis = 0), nrep, axis = 1)
    # look up datafiles
    directory = "../dat/Normal_hyperspectral_images"
    datafiles = bop_analysis.LookUpData(open(directory + ".txt", "r"), lookin) 

    # for each patch  
    for pa in pas:

        # load the picked data
        dat = bop_analysis.LoadGather(pa, directory = directory)

        # define sets of 4 each, defining which images are the same specimen
        maskset = [[8,9,10,11], [12,13,14,15],[16,17,18,19], [20,21,22,23], [0,1,2,3], [4,5,6,7],]

        # if we're only plotting half the specimens
        if halfSpecimens == True:
            fig, axs = plt.subplots(nrows=3, ncols=4, dpi = 300, figsize = (5,10),
                           squeeze=True)    
            specimens = np.arange(0,6,2)
        else:
            fig, axs = plt.subplots(nrows=4, ncols=6, dpi = 300, figsize = (10,15),
                           squeeze=True)
            specimens = np.arange(0,6)
           
        for sett in specimens: # for each specimen

            for k in range(0,4): # for each set of 4 views
                
                # load masking file
                mat_contents = sio.loadmat(datafiles["matfiles"][maskset[sett][k]], struct_as_record=False, squeeze_me=True)
                mat_contents = mat_contents['ImStruct']

                # work out specimen and view
                thisSpecimen, thisView = GetSpecimenID(datafiles["matfiles"][maskset[sett][k]])
                thisSpecies = thisSpecimen[:-3]

                # make a logical mask for the sampling data on basis of specimen and view
                datmask = np.logical_and(dat['meta']['specimen'] == thisSpecimen, dat['meta']['facing'] == thisView)

                # get hyperspectral image
                img = envi.open(datafiles["hdrfiles"][maskset[sett][k]], datafiles["bilfiles"][maskset[sett][k]])

                if hasattr(mat_contents, 'bwlist'):
                    masknames = mat_contents.bwlist
                    # print(masknames)

                    # get whole mask
                    mask = mat_contents.masklist[0]
                    mask = mask[::scale_factor, ::scale_factor]

                    # find where mask starts and stops in x and y, just for nice plotting limits
                    buff = 25
                    x = np.where(mask.sum(axis=0)>=1) # get 1's as columns
                    y = np.where(mask.sum(axis=1)>=1) # get 1's as rows
                    x = [np.min(x) - buff, np.max(x) + buff]
                    y = [np.max(y) + buff, np.min(y) - buff]

                    # get subplot
                    if halfSpecimens == True:
                        ax = axs[np.argwhere(specimens==sett)[0][0],k]
                    else:
                        ax = axs[k,np.argwhere(specimens==sett)[0][0]]

                    if pa < 0: # skip labelling on whole
                        if k == 0:
                            ax.set_title(thisSpecimen)
                        if sett == 0:
                            ax.set_ylabel(thisView)

                    # RGB image
                    _, _, rgb_im = utils.PlotFalseRGB(img, r = 600., g = 550., b = 425., standard_100 = mat_contents.ref99, normalize_max = .3)
                    ax.imshow(np.dstack((rgb_im[::scale_factor, ::scale_factor, :], mask)), alpha = mask)

                    ma = mat_contents.masklist[pa]
                    if ~np.isnan(ma).all():
                        expanded = binary_dilation(ma, structure = bin_struc, iterations = 5) # dilation of mask
                        ax.imshow(np.ones(ma[::scale_factor, ::scale_factor].shape), alpha = expanded[::scale_factor, ::scale_factor] - ma[::scale_factor, ::scale_factor], cmap = 'tab10') # subtracting mask from dilated mask gives difference (outline)
                    
                    ax.xaxis.set_ticks([])
                    ax.yaxis.set_ticks([])
                    ax.xaxis.set_ticklabels([])
                    ax.yaxis.set_ticklabels([])
                    # remove box
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    # set limits
                    ax.set_xlim(x)
                    ax.set_ylim(y)

                    # add text number of samples
                    if not halfSpecimens:
                        tx = ax.text(np.mean(x), np.max(y), np.sum(datmask), ha='right', rotation = 0)
                        # tx.set_path_effects(effects)

                    col = color_dict.get(thisSpecies)
                    if halfSpecimens == True:
                        if pa==0:
                            ax.scatter(dat['meta']['x'][datmask]/scale_factor, dat['meta']['y'][datmask]/scale_factor, s = 3.25, color = col, edgecolors=None)
                        else:
                            ax.scatter(dat['meta']['x'][datmask]/scale_factor, dat['meta']['y'][datmask]/scale_factor, s = 0.5, color = col, edgecolors=None)
                    else:
                        if pa==0:
                            ax.scatter(dat['meta']['x'][datmask]/scale_factor, dat['meta']['y'][datmask]/scale_factor, s = 4, color = col, edgecolors=None)
                        else:
                            ax.scatter(dat['meta']['x'][datmask]/scale_factor, dat['meta']['y'][datmask]/scale_factor, s = 1, color = col, edgecolors=None)

        fig.subplots_adjust(hspace=0, wspace = 0)

        if saveplot == True:
            if halfSpecimens == True:
                fn = r'../out/Figures/SampleCheck/' + fname + mat_contents.bwlist[pa] + '_half'
            else:
                fn = r'../out/Figures/SampleCheck/' + mat_contents.bwlist[pa] 
            fig.savefig(fname = fn + '.png', dpi = 150, bbox_inches='tight', transparent=True)
            fig.savefig(fname = fn + '.pdf', dpi = 300, bbox_inches='tight', transparent=True)
            
    return ax
    
def SampleCheck_breast(lookin, color_dict = dict({'CM_M':[230/255,159/255,0/255],
                  'CR_M':[0/255,158/255,115/255],
                  'H_M': [86/255,180/255,233/255]}), saveplot = True, scale_factor = 4):

    """ As SampleCheck_half, but only plotting breast samples in the tilted orientation. This is a very memory intensive function. """

    directory = "../dat/Tilted_hyperspectral_images"
    datafiles = bop_analysis.LookUpData(open("../dat/Tilted_hyperspectral_images.txt", "r"), lookin) 
    
    # dilation structure
    nrep = 5 # in combo with iterations = 5, boundary of 25 pixels
    bin_struc = np.repeat(np.repeat(generate_binary_structure(2, 1), nrep, axis = 0), nrep, axis = 1)
    
    # load appropriate patch data       
    pa = 2
    dat = bop_analysis.LoadGather(pa, directory = directory)

    # sets of 4 each
    maskset = [2, 3, 4, 5, 0, 1]

    fig, axs = plt.subplots(nrows=1, ncols=6, dpi = 150, figsize = (10,15/4),
                       squeeze=True)

    specimens = np.arange(0,6)

    for sett in specimens: # for each specimen
        
        # load masking file
        mat_contents = sio.loadmat(datafiles["matfiles"][maskset[sett]], struct_as_record=False, squeeze_me=True)
        mat_contents = mat_contents['ImStruct']

        # work out specimen and view
        thisSpecimen, thisView = GetSpecimenID(datafiles["matfiles"][maskset[sett]])
        thisSpecies = thisSpecimen[:-3]

        # make a logical mask for the sampling data on basis of specimen and view
        datmask = np.logical_and(dat['meta']['specimen'] == thisSpecimen, dat['meta']['facing'] == thisView)

        # get hyperspectral image
        img = envi.open(datafiles["hdrfiles"][maskset[sett]], datafiles["bilfiles"][maskset[sett]])

        if hasattr(mat_contents, 'bwlist'):
            masknames = mat_contents.bwlist

            # get whole mask
            mask = mat_contents.masklist[0]
            mask = mask[::scale_factor, ::scale_factor]

            # find where mask starts and stops in x and y, just for nice plotting limits
            buff = 25
            x = np.where(mask.sum(axis=0)>=1) # get 1's as columns
            y = np.where(mask.sum(axis=1)>=1) # get 1's as rows
            x = [np.min(x) - buff, np.max(x) + buff]
            y = [np.max(y) + buff, np.min(y) - buff]

            # get subplot
            ax = axs[np.argwhere(specimens==sett)[0][0]]

            if pa < 0: # skip labelling on whole for now, for plotting with
                if k == 0:
                    ax.set_title(thisSpecimen)

                if sett == 0:
                    ax.set_ylabel(thisView)

            # RGB image underneath    
            _, _, rgb_im = utils.PlotFalseRGB(img, r = 600., g = 550., b = 425., standard_100 = mat_contents.ref99, normalize_max = .3)
            ax.imshow(np.dstack((rgb_im[::scale_factor, ::scale_factor, :], mask)), alpha = mask)

            ma = mat_contents.masklist[pa]
            if ~np.isnan(ma).all():
                expanded = binary_dilation(ma, structure = bin_struc, iterations = 5) # dilation of mask
                ax.imshow(np.ones(ma[::scale_factor, ::scale_factor].shape), alpha = expanded[::scale_factor, ::scale_factor] - ma[::scale_factor, ::scale_factor], cmap = 'tab10') # subtracting mask from dilated mask gives difference (outline)

            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            # remove box
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            # set limits
            ax.set_xlim(x)
            ax.set_ylim(y)

            tx = ax.text(np.mean(x), np.max(y), np.sum(datmask), ha='right', rotation = 0)
            # tx.set_path_effects(effects)

            col = color_dict.get(thisSpecies)

            if pa==0:
                ax.scatter(dat['meta']['x'][datmask]/scale_factor, dat['meta']['y'][datmask]/scale_factor, s = 4, color = col, edgecolors=None)
            else:
                ax.scatter(dat['meta']['x'][datmask]/scale_factor, dat['meta']['y'][datmask]/scale_factor, s = 1, color = col, edgecolors=None)

    fig.subplots_adjust(hspace=0, wspace = 0)

    if saveplot == True:
        fn = r'../out/Figures/SampleCheck/' + 'tilted_' + mat_contents.bwlist[pa] 
        fig.savefig(fname = fn + '.png', dpi = 300, bbox_inches='tight', transparent=True)
        fig.savefig(fname = fn + '.pdf', dpi = 300, bbox_inches='tight', transparent=True)
        
    return ax

def GetSpecimenID(fn):

    """ Utility function to get specimen ID from name, used in samplecheck """

    # fn = os.path.basename(datafiles["matfiles"][maskset[sett][k]])
    fn = os.path.basename(fn)
    fn = re.split('(\d+)', fn)
    thisSpecimen = fn[0] + fn[1]
    thisView = fn[2].replace('_', '')
    thisView = thisView.replace('.mat', '')
    print(thisSpecimen)
    print(thisView)
    return thisSpecimen, thisView
    

def ScottFactorKDE(pltd):

    """ Utility function to generate scott factor across a dataset for 3d KDEs (to use the same kernel bandwidth across species in plots, rather than 
    calculating a new bandwidth on only the subsample). As per https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html, 
    bw estimation
    """
    
    n = pltd.shape[0]
    d = pltd.shape[1]
    f = n**(-1./(d+4))
    print(f)
    return f
    
def tcs_multi(label_dict, color_dict, saveplot = True):

    """ Multi-panel 3D Tetraedral XYZ plots in plotly """

    maskin = [0, 0, 2, 3, 4, 6]
    includeWhole = 1

    fig = make_subplots(
    rows=2, cols=6,
    column_widths=[2, 1, 1, 1, 1, 1],
    horizontal_spacing = 0,
    vertical_spacing = 0,
    specs=[[{"rowspan":2, "type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
           [None, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
    # subplot_titles=mat_contents.bwlist.tolist(),
    print_grid=False)

    # for each patch
    for i in range(len(maskin)):
        print('Working on ' + str(i+1) + ' of ' + str(len(maskin)))

        # load the relevant pickled data
        if maskin[i]==2: # for breast, use tilted image samples instead
            dat = bop_analysis.LoadGather(maskin[i], directory = '../dat/Tilted_hyperspectral_images/')
        else:
            dat = bop_analysis.LoadGather(maskin[i], directory = '../dat/Normal_hyperspectral_images/')
            
        # since we're plotting some colors, can re-generate RGB, this allows us to dynamically
        # update color scaling used for plotting if wanted.
        # XYZ, sRGB, Lab = utils.GetColors(dat['arr'], dat['wl'])
        sRGB = dat['sRGB']

        # choose relevant distances etc
        pltn = 'txyz'
        pltd =  dat[pltn]

        # plot the tetrahedron, and color it
        tetra.tetra3d(fig, row = 1, col = i+1, add_labels = i == 0, labels = ['VS', 'SWS', 'MWS', 'LWS'], label_size = 22)

        # scatter all samples in the tetrahedron
        fig.add_trace(go.Scatter3d(x=pltd[:,0], y=pltd[:,1], z=pltd[:,2], 
                      mode='markers', 
                      marker=dict(size=3, color = sRGB)
                     ), row = 1, col = i+1) 

        if i > 0: # work out KDEs for patches after the first

            sf = ScottFactorKDE(pltd) # use same scale factor for KDE across each species
            tetra.tetra3d(fig, opacity = 0, row = 2, col = i+1, add_labels = False)

            for spp in np.unique(dat['meta']['species']):

                wcolor = color_dict.get(spp)
                wcolor = 'rgb(' + ', '.join(str(e) for e in wcolor) + ')'

                # Calculate density, change res to speed up/slow down, 10 is fast, 60 is slow
                xi, yi, zi, density = tetra.Get3dKDE(pltd[dat['meta']['species']==spp], res = 50, bw_method = sf)
                density = density/density.max()
                density.max()

    #             fig.add_trace(go.Volume( # Alternatively, use volume rather than a 3d isosurface, more accurate, but for my purposes too busy
    #             x=xi.flatten(),
    #             y=yi.flatten(),
    #             z=zi.flatten(),
    #             colorscale = [[0, wcolor], [1,wcolor]],
    #             value=density.flatten(),
    #             isomin=0.001,
    #             isomax=0.5,
    #             opacity=0.9, # max opacity # needs to be small to see through all surfaces
    #             surface_count=3, # needs to be a large number for good volume rendering
    #             name = None,
    #             showlegend=True,
    #             showscale=False,
    #             ), row = 2, col = i + 1)

                fig.add_trace(go.Isosurface(
                x=xi.flatten(),
                y=yi.flatten(),
                z=zi.flatten(),
                value=density.flatten(),
                colorscale = [[0, wcolor], [1,wcolor]],
                isomin=0.05,
                isomax=0.05,
                surface_count=1,
                opacity = 0.5,
                showscale = False,
                caps=dict(x_show=False, y_show=False)
                ), row = 2, col = i + 1)


    # set camera angle, save

    camera2 = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.45, y=1.2, z=1.05),
        projection = dict(type = 'orthographic')
    )

    camera2_zoom = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0.3389370281166833, y=-0.18528491624594304, z=-0.25630170597529356),
        eye=dict(x=0.8187698345557777*1.45, y=0.21181809597951431*1.45, z=0.09116342972198166*1.1),
        projection = dict(type = 'perspective') # eye distance doesn't seem to do anything when view is orthographic!
    )

    fig.update_scenes(camera = camera2, 
                      xaxis = dict(showbackground=False, visible=False),
                      yaxis = dict(showbackground=False, visible=False),
                      zaxis = dict(showbackground=False, visible=False))

    # update the smaller plots independently
    for i in range(2, 8):
        fig.update_scenes(camera = camera2_zoom, row = 1, col = i)
        fig.update_scenes(camera = camera2_zoom, row = 2, col = i)

    fig.update_layout(showlegend=False, paper_bgcolor='rgba(255,255,255,1)', plot_bgcolor='rgba(255,255,255,1)',
                     margin = dict(t = 0, b = 0, l = 0, r = 0),
                     height=400, width=1200)

    if saveplot == True:
        try:
            fig.write_image('../out/Figures/TCS_multi.png', scale=8, engine='kaleido')
        except:
            print('TCS_multi: Plotly static figure export/kaleido is not working correctly, try rolling back kaleido. Figure may still be generated, just not saved.')
    
    return fig


def RNL_multi(label_dict, color_dict, saveplot = True):
    
    """ Multi-panel 3D plot of RNL space in plotly """

    maskin = [0, 2, 3, 4, 6] 
    includeWhole = 1

    fig = make_subplots(
    rows=2, cols=6,
    column_widths=[2, 1, 1, 1, 1, 1],
    horizontal_spacing = 0,
    vertical_spacing = 0,
    specs=[[{"rowspan":2, "type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
           [None, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
    subplot_titles=['Whole', 'Breast', 'Belly/vent', 'Shoulder', 'Back', '', '', '', '', ''], 
    print_grid=False)


    for i in range(len(maskin)):

        print('Working on ' + str(i+1) + ' of ' + str(len(maskin)))

        # load the relevant data
        if maskin[i]==2: # for breast, use tilted image samples instead
            dat = bop_analysis.LoadGather(maskin[i], directory = '../dat/Tilted_hyperspectral_images/')
        else:
            dat = bop_analysis.LoadGather(maskin[i], directory = '../dat/Normal_hyperspectral_images/')
            
        # since we're plotting some colors, can re-generate RGB, this allows us to dynamically
        # update color scaling used for plotting if wanted.
        # XYZ, sRGB, Lab = utils.GetColors(dat['arr'], dat['wl'])
        sRGB = dat['sRGB']

        # apply RNL model with vision parameters
        cone_ratio = np.array((1,1,1,2)) # columbia livia (Vorobyev & Osorio 1998)
        weber_ratio = 0.05 # 0.1 is often used, 0.05 is more conservative
        x, y, z, _= utils.GetRNL(dat['relusml'], ratios = cone_ratio, weber_ratio = weber_ratio, logarithm = True)
        pltd = np.vstack((x,y,z)).T

        # plot for big axis
        fig.add_trace(go.Scatter3d(x=pltd[:,0], y=pltd[:,1], z=pltd[:,2], 
                      mode='markers', 
                      marker=dict(size=3, color = sRGB)
                     ), row = 1, col = i+1) 

        # for smaller subplots
        if i > 0:

            sf = ScottFactorKDE(pltd) # find combined KDE kernel size across species to avoid bias
            for spp in np.unique(dat['meta']['species']):

                wcolor = color_dict.get(spp)
                wcolor = 'rgb(' + ', '.join(str(e) for e in wcolor) + ')'

                # Calculate density, change res to speed up/slow down 10 is fast, 60 pretty fine
                xi, yi, zi, density = tetra.Get3dKDE(pltd[dat['meta']['species']==spp], res = 50, bw_method = sf)
                density = density/density.max()
                density.max()

                fig.add_trace(go.Isosurface(
                x=xi.flatten(),
                y=yi.flatten(),
                z=zi.flatten(),
                value=density.flatten(),
                colorscale = [[0, wcolor], [1,wcolor]],
                isomin=0.05,
                isomax=0.05,
                surface_count=1,
                opacity = 0.5,
                showscale = False,
                caps=dict(x_show=False, y_show=False)
                ), row = 2, col = i + 1)


    # set camera angle, save
    camera2_zoom = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.7, y=1.7, z=1.7),
        projection = dict(type = 'perspective') # eye distance doesn't seem to do anything when view is orthographic!
    )

    fig.update_scenes(camera = camera2_zoom)

    # update the smaller ones independently
    for i in range(2, 8):
        fig.update_scenes(camera = camera2_zoom, row = 1, col = i, 
                          xaxis = dict(nticks=3), 
                          yaxis = dict(nticks=3), 
                          zaxis = dict(nticks=3))
        fig.update_scenes(camera = camera2_zoom, row = 2, col = i, 
                          xaxis = dict(nticks=3), 
                          yaxis = dict(nticks=3),
                          zaxis = dict(nticks=3))

    fig.update_layout(showlegend=False, paper_bgcolor='rgba(255,255,255,1)', plot_bgcolor='rgba(255,255,255,1)',
                     margin = dict(t = 50, b = 50, l = 50, r = 50),
                     height=400, width=1200, font=dict(family = "Times New Roman"))

    if saveplot == True:
        try:
            fig.write_image('../out/Figures/RNL_multi_' + 'weber' + str(weber_ratio) + 'n' + ''.join(map(str, cone_ratio)) + '.png', scale=8, engine='kaleido')
        except:
            print('RNL_multi: Plotly static figure export/kaleido is not working correctly, try rolling back kaleido. Figure may still be generated, just not saved.')
    
    return fig

def montage(img, mat_contents, nrow = 2, ncol = 10, minwl = 330, maxwl = 710, 
    x_sam = [500, 1000, 1350, 2000], y_sam = [370, 370, 370, 370], 
    saveplot = False):

    """ plot of a montage of slices from one hsi, and optionally a plot of the spectra close to some sampled points (x_sam, y_sam) """

    # Find limits of mask,
    mask = mat_contents.masklist[0] # indicates the 'whole' mask

    # find where mask starts and stops in x and y, just for nice plotting limits a little later on
    x_lim = np.where(mask.sum(axis=0)>=1) # get 1's as columns
    y_lim = np.where(mask.sum(axis=1)>=1) # get 1's as rows
    x_lim = [np.min(x_lim), np.max(x_lim)]
    y_lim = [np.min(y_lim), np.max(y_lim)]
    
    # generate a nice rainbow colormap, now in utils
    cmap = utils.GetCustomColormap()

    # get prepared for montage
    owl = img.bands.centers
    nwl = nrow * ncol
    imw = x_lim[1]-x_lim[0]
    imh = y_lim[1]-y_lim[0]
    wwidth = ncol * (imw/imh)
    wheight = nrow
    wwidth = wwidth/wheight
    wheight = wheight/wheight
    levs = np.linspace(minwl, maxwl, nwl)
    print(levs)
    idxl = []
    for i in levs:
        idxl.append(np.argmin(np.abs(owl - i)))
    rgba = cmap(np.linspace(0,1,len(levs)))
    
    # Also define some samples to overplot, and extract
    x_sam = x_sam + y_lim[0] # sample (in terms of pix in image), note that x and y are swapped in plot
    y_sam = y_sam + x_lim[0]

    cols = ['orange', 'chocolate', 'magenta', 'purple']
    nams = ['a', 'b', 'c', 'd']
    
    fig, axs = plt.subplots(nrow, ncol, figsize=[wwidth * 8,wheight * 8], dpi=100, facecolor='w', edgecolor='k') # n by n*3.38
    for i, ax in enumerate(fig.axes):
    
        print('Working on ' + str(i) + ' of ' + str(nrow * ncol))
        
        # divide relevant band by standard to generate reflectance
        gray_img = np.divide(img[:,:,int(idxl[i])], mat_contents.ref99[idxl[i]])

        # plot band
        ax.imshow(gray_img, cmap='gray', vmin = 0, vmax = 1)

        # add colored rectangle
        rect = plt.Rectangle((0,y_lim[1]-400), gray_img.shape[1], 400, facecolor=rgba[i])
        ax.add_patch(rect)
        # and display wavelength
        ax.text(x_lim[0]+((x_lim[1]-x_lim[0])/2), y_lim[1]-200,  str(levs[i].astype(int)) + " nm", horizontalalignment='center', verticalalignment='center', size = 20)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # limit to mask
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[1], y_lim[0])
        ax.set_aspect('equal')
        for k in range(0,4):
            ax.scatter(y_sam[k], x_sam[k], s = 50)
    plt.subplots_adjust(wspace=0, hspace=0)

    if saveplot == True: 
        fig.savefig('../out/Figures/Diagram/Montage_plot.pdf', dpi = 200,bbox_inches='tight')
        
    ## Then, also plot spectra around the samples defined there
    ex = 10 # extent around sample to read

    # plot lines
    fig, ax = plt.subplots(1)

    for i in range(len(x_sam)):
        print(i)
        Specs = utils.ReadBands(img, np.arange(y_sam[i]-ex, y_sam[i]+ex), np.arange(x_sam[i]-ex, x_sam[i]+ex), 5)
        
        # interp and extrap wl
        plt_wl, plt_Spec = utils.InterpExtrapSample(img.bands.centers, Specs, window_length = 45)
        wl_mask = plt_wl<325
        
        # get colors too
        plt_XYZ, plt_sRGB, plt_Lab = utils.GetColors(plt_Spec, plt_wl)

        # plot lines
        for p in range(plt_Spec.shape[0]):
            ax.plot(plt_wl[~wl_mask], plt_Spec[p,~wl_mask].T, lw = 4, c = plt_sRGB[p,:], alpha = 0.1, label=nams[i])    

        # plot mean lines
        ln = ax.plot(plt_wl[~wl_mask], np.mean(plt_Spec[:,~wl_mask].T, axis = 1), lw = 2, c = 'black', alpha = 1, label=nams[i])
        
        # plot interpolated part
        ax.plot(plt_wl[wl_mask], plt_Spec[:,wl_mask].T, lw = 4, linestyle='dotted')
        
        ypoint=(np.mean(plt_Spec[:,~wl_mask].T, axis = 1)[-1])
        ax.scatter([710], ypoint, s = 70)

    ax.vlines(325, 0, 1.2, colors='Black', linestyles='dashed', zorder=3)
    ax.axes.set_xlabel('Wavelength (nm)')
    ax.axes.set_ylabel('Reflectance')
    ax.axes.set_ylim([0,0.8])
    
    fig.tight_layout()

    if saveplot == True: 
        fig.savefig('../out/Figures/Diagram/Spectra_plot.pdf', dpi = 100)

def Hyper_render(img, mat_contents, x_sam = [500, 1000, 1350, 2000], y_sam = [370, 370, 370, 370], saveplot = False, normalize_max = .2):

    """ Save a falsecolor RGB image of a hsi where we use mask 'whole' to make the background black and scatter some sample locations over the top """

    # Find limits of mask,
    mask = mat_contents.masklist[0] # indicates the 'whole' mask

    # find where mask starts and stops in x and y, just for nice plotting limits
    x_lim = np.where(mask.sum(axis=0)>=1) # get 1's as columns
    y_lim = np.where(mask.sum(axis=1)>=1) # get 1's as rows
    x_lim = [np.min(x_lim), np.max(x_lim)]
    y_lim = [np.min(y_lim), np.max(y_lim)]
    
    # Also define some samples to overplot, and extract
    x_sam = x_sam + y_lim[0] # sample (in terms of pix in image), note that x and y are swapped in plot
    y_sam = y_sam + x_lim[0]

    # plot quite nicely
    fig, ax = plt.subplots(1, figsize=(12,12), dpi=100, facecolor='w', edgecolor='k')

    buff = 200
    _, _, rgb_im = utils.PlotFalseRGB(img, r = 600., g = 550., b = 425., standard_100 = mat_contents.ref99, normalize_max = normalize_max)
    ax.imshow(np.dstack((rgb_im, mask[:,:,np.newaxis])), alpha=mask>0)
    # limit to mask
    ax.set_xlim(x_lim[0]-buff, x_lim[1]+buff)
    ax.set_ylim(y_lim[1]+buff, y_lim[0]-buff)
    # plot sample locations
    for k in range(0,4):
        ax.scatter(y_sam[k], x_sam[k], s = 50)
    ax.set_facecolor('xkcd:black')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False,
                   left=False, labelleft=False)

    if saveplot == True: 
        fig.savefig('../out/Figures/Diagram/Hyper_render.png', dpi = 400, bbox_inches='tight', pad_inches=0)
        

def UDLR_render(img, mat_contents, ud, lr, saveplot = False, normalize_max = .2):

    """ Generate falsecolor RGB version of hsi, and next to it plot the up-down and left-right surface normal images for that hsi, cropped """

    # Find limits of mask,
    mask = mat_contents.masklist[0] # indicates the 'whole' mask

    # find where mask starts and stops in x and y, just for nice plotting limits
    x_lim = np.where(mask.sum(axis=0)>=1) # get 1's as columns
    y_lim = np.where(mask.sum(axis=1)>=1) # get 1's as rows
    x_lim = [np.min(x_lim), np.max(x_lim)]
    y_lim = [np.min(y_lim), np.max(y_lim)]

    wwidth = x_lim[1]-x_lim[0]
    wheight = y_lim[1]-y_lim[0]
    wheight = wheight/wwidth
    wwidth = wwidth/wwidth
    
    _, _, rgb_im = utils.PlotFalseRGB(img, r = 600., g = 550., b = 425., standard_100 = mat_contents.ref99, normalize_max = normalize_max)

    fig, ax = plt.subplots(1, 3, figsize = (wwidth * 10, wheight * 10))

    i1 = ax[0].imshow(ud[y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]])
    # fig.colorbar(i1, ax=ax[0])

    i2 = ax[1].imshow(lr[y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]])
    # fig.colorbar(i2, ax=ax[1])

    temp_im = np.dstack((rgb_im, mask[:,:,np.newaxis]))
    temp_im = temp_im[y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]

    i3 = ax[2].imshow(temp_im)

    buff = 100
    ax[2].axis('off')
    ax[0].axis('off')
    ax[1].axis('off')

    # fig.tight_layout()
    plt.subplots_adjust(wspace=.25, hspace=0)

    if saveplot == True: # only overwrite this image if we're using example 1
        fig.savefig('../out/Figures/Diagram/Normals_plot_2.pdf', dpi = 300,bbox_inches='tight')
        fig.savefig('../out/Figures/Diagram/Normals_plot_2.png', dpi = 300,bbox_inches='tight')
        
def TailPlot(label_dict, color_dict, pltn = 'KLPD_umap_embedded', max_point_manual = None, saveplot = True):

    """ Generate a plot of the colors of the tail embedded with in PCA, KLPD or dSH distances """

    # Load tail example for plotting
    # Note that this runs on picked analyzed data, so must run after bop_analysis.RunDataAnalysis()
    k = 0
    directory = '../dat/Tail_hyperspectral_images'
    dat = bop_analysis.LoadGather(k, directory = directory)
    bgcolor = '#5A5A5A'
        
    # Make plot
    fig = plt.figure(figsize = (16,5))
    ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=1)
    ax2 = plt.subplot2grid((1, 4), (0, 1), colspan=1)
    ax3 = plt.subplot2grid((1, 4), (0, 2), colspan=2)

    pltd = dat[pltn]

    i = 0

    # sRGB plot for first row
    ax1.scatter(x = pltd[:,0], 
                    y = pltd[:,1], 
                    c = dat['sRGB'], 
                    alpha=1)
    ax1.set_facecolor(bgcolor)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # get limits
    left, right = ax1.get_xlim()
    bottom, top = ax1.get_ylim()

    # kde plot
    sns.kdeplot(x = pltd[:,0], 
            y = pltd[:,1], 
            hue = [label_dict[species] for species in dat['meta']['species']], # relabel
            palette = dict((label_dict[key], value) for (key, value) in color_dict.items()),
            fill=False,
            thresh=0, 
            common_norm = False, 
            legend = False,
            levels=[0.2], # [0.2, 0.6, 0.8],
            ax=ax2)
    ax2.set_facecolor(bgcolor)
    ax2.set_xlim(left,right)
    ax2.set_ylim(bottom, top)
    ax2.set_xticks([])
    ax2.set_yticks([])

    listsp = np.asarray([label_dict[species] for species in dat['meta']['species']])
    thing = np.unique(listsp)
    print(thing)

    for x, zz in zip(thing, range(len(thing))):
        # get x and y embedded pos for this species point
        kx, ky = pltd[listsp==x,0], pltd[listsp==x,1]

        if max_point_manual is None: # if looking for maximum density
            # lims for kernel
            xmin, xmax = np.min(pltd[:,0]), np.max(pltd[:,0])
            ymin, ymax = np.min(pltd[:,1]), np.max(pltd[:,1])
    
            # generate kernel to find area of max density
            X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            # apply kernel to x and y from species
            kernel = gaussian_kde(np.vstack([kx, ky]))
            Z = np.reshape(kernel(positions).T, X.shape)
            # predict with kernel all values including other species 
            # (to get back to original array shape), here we could also only
            # plot spectra from that species?
            val_pdf = kernel.pdf(pltd.T)
            val_pdf[listsp!=x]=float('-inf')
    
            # find max value
            max_point = np.argmax(val_pdf)
            max_loc =  pltd[max_point]
            print(max_loc)

        else: # if we manually selected points of interest (1 for each species)
            if len(thing) != max_point_manual.shape[0]:
                raise Exception("Length of max_point_manual must equal number of species")
            max_loc = max_point_manual[zz, :]

        # find distance of points to our selected point
        n_spec = 50
        curr_point_dist = np.linalg.norm(np.subtract(pltd, max_loc), axis = 1)

        # record the closest n idx's and closest location
        closest_point_idx = curr_point_dist.argsort()[:n_spec]
        closest_point_xy = pltd[curr_point_dist.argsort()[:1],:]

        # get the spectra for those idxes for plotting and averaging
        y_plot = dat['arr'][closest_point_idx,:].T 
        x_plot = dat['wl']

        # get median and mad
        me_plot = np.median(y_plot, axis = 1)
        # sd_plot = np.std(y_plot[:,listsp==x], axis = 1)
        sd_plot = median_abs_deviation(y_plot, axis = 1)

        # get species color
        scol = dict((label_dict[key], value) for (key, value) in color_dict.items()).get(x)

        # scatter markers where the highest density is
        ax2.scatter(closest_point_xy[0][0], closest_point_xy[0][1], s = 50, color = scol, 
                                         edgecolors = 'black', linewidths = 1)      

        wl_mask = x_plot<325

        # get mean colors too
        plt_XYZ, plt_sRGB, plt_Lab = utils.GetColors(me_plot, x_plot)
        plt_sRGB = scol

        # plot mean and sd
        ln = ax3.plot(x_plot[~wl_mask], me_plot[~wl_mask], color = plt_sRGB, linewidth = 1.5, label = x)
        ax3.fill_between(x_plot[~wl_mask], me_plot[~wl_mask]-sd_plot[~wl_mask], me_plot[~wl_mask]+sd_plot[~wl_mask], color = plt_sRGB, alpha = 0.5)
        ax3.plot(x_plot[wl_mask], me_plot[wl_mask], color = plt_sRGB, linewidth = 1.5, linestyle = 'dotted')

    ax3.set_facecolor(bgcolor)
    ax3.set_ylabel('Reflectance (0-1)')
    ax3.set_xlabel('Wavelength (nm)')
    
    fig.tight_layout()

    if saveplot == True:
        fig.savefig('../out/Figures/' + 'Tail_' + pltn + '.png', dpi = 300, bbox_inches="tight")
        fig.savefig('../out/Figures/' + 'Tail_' + pltn + '.pdf', dpi = 300, bbox_inches="tight")

    return fig
    
def AnglePlots(saveplot = True):

    """ For the breast patch, plot color against angle (from udlr) in a few different ways """

    # load the relevant data
    k = 2 # breast mask
    dat = bop_analysis.LoadGather(k, directory = '../dat/Tilted_hyperspectral_images/')
    wl = dat['wl']
    arr = dat['arr']
    lr = dat['meta']['listlr']
    ud = dat['meta']['listud']
    sRGB = dat['sRGB']

    # also load non-tilted ('normal') breast mask data, concatenate all these data
    dat = bop_analysis.LoadGather(k, directory = '../dat/Normal_hyperspectral_images/')
    arr = np.vstack((arr, dat['arr']))
    wl = dat['wl']
    lr = np.hstack((lr, dat['meta']['listlr']))
    ud = np.hstack((ud, dat['meta']['listud']))
    sRGB_2 = dat['sRGB']
    sRGB = np.vstack((sRGB, sRGB_2))

    ## Scatter the points by their normal angle, and bin the vertical angle
    lims = np.linspace(np.pi,-np.pi,4)
    dec_place = 2
    x1 = -np.pi
    x2 = np.pi
    y1 = lims
    y2 = lims

    fig, ax = plt.subplots(1, figsize = (5,5))
    ax.scatter(lr[np.argsort(np.sum(sRGB, axis = 1))], ud[np.argsort(np.sum(sRGB, axis = 1))], 
               c = sRGB[np.argsort(np.sum(sRGB, axis = 1)),:], s = 20);
    ax.set_facecolor('xkcd:black')

    ax.set_prop_cycle(plt.cycler('color', ['c', 'm', 'r']) +
                       plt.cycler('linestyle', ['-', '--', ':']))

    for i in lims:
        plt.plot([x1, x2], [i, i], color = 'white', linestyle='dashed', linewidth=1)
        
    ax.set_prop_cycle(plt.cycler('color', ['c', 'm', 'r']) +
                       plt.cycler('linestyle', ['-', '--', ':']))

    for j, i in enumerate(lims[0:-1]):
        me_t = str(round(lims[j], dec_place)) + ' to ' + str(round(lims[j+1], dec_place))
        plt.text(np.pi, i - (lims[0]-lims[1])/2, s = me_t, ha = 'right', c = 'white')
        
    ax.set_xlabel('Horizontal angle (-π to π)')
    ax.set_ylabel('Vertical angle (-π to π)')
        
    # save figure
    fig.savefig('../out/Figures/' + 'Breast_angle_both.png', dpi = 300, bbox_inches="tight")
    fig.savefig('../out/Figures/' + 'Breast_angle_both.pdf', dpi = 300, bbox_inches="tight")
    
    #### plot the median and mad for each of the vertical bins in the above figure
    
    me_plot = np.zeros(( len(lims)-1, arr.shape[1]))
    se_plot = np.zeros(( len(lims)-1, arr.shape[1]))
    me_t = np.empty(len(lims)-1, dtype=object)

    for i in range(len(lims)-1):
        
        if i<len(lims)-1:
            print(lims[i])
            
            logmask = np.logical_and(np.logical_and(ud<lims[i], ud>lims[i+1]), np.sum(arr,axis=1) > 2) # not very dark spectra
            if(np.sum(logmask)>0):
                y_plot = arr[logmask]
                x_plot = dat['wl']

                # get median and mad for interval
                me_plot[i] = np.median(y_plot, axis=0) # mean and median both look similar
                se_plot[i] = median_abs_deviation(y_plot)
                me_t[i] = str(round(lims[i], dec_place)) + ' to ' + str(round(lims[i+1], dec_place))
                

    fig, ax = plt.subplots(1)
    ax.set_prop_cycle(plt.cycler('color', ['k', 'k', 'k','k', 'k', 'k','k', 'k', 'k']) +
                       plt.cycler('linestyle', ['-', '-','-','--', '--','--',':',':',':']))
                       
    for i in range(me_plot.shape[0]):
        
        ax.plot(dat['wl'], me_plot[i,:], label = me_t[i]);
        ax.plot(dat['wl'], me_plot[i,:] - se_plot[i,:], linewidth = 0.5);
        ax.plot(dat['wl'], me_plot[i,:] + se_plot[i,:], linewidth = 0.5);
        ax.fill_between(dat['wl'], me_plot[i,:] - se_plot[i,:], me_plot[i,:] + se_plot[i,:], alpha=0.15);
        
    ax.legend();
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance (0-1)')

    # save figure
    if saveplot == True:
        fig.savefig('../out/Figures/' + 'Breast_angle_spectra.png', dpi = 300, bbox_inches="tight")
        fig.savefig('../out/Figures/' + 'Breast_angle_spectra.pdf', dpi = 300, bbox_inches="tight")

