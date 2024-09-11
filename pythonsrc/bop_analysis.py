## Functions specific to the BOP data-set 

## ANALYSIS ##

from scipy import io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
import spectral.io.envi as envi 
import time
import re

import hsi.tetra as tetra
import hsi.utils as utils
import hsi.triangle as triangle
import bop_sampling
import bop_plotting

def RunDataAnalysis(lookin, udlr_dir, force = True, mainplots = True, additionalplots = False, label_dict = None, color_dict = None, color_scale = 1):
    """ 
    Function to run the data gathering process easily for each of the 3 sets of HSI we have (normal, tilted, and tail), 
    and for each, samples out a bunch of spectra, it also samples from the surface normal images produced from 3D modelling ("udlr") image if it exists
    Then it embeds those spectra in various ways and pickles (saves it to a file) for ease of use in the later analysis

    lookin (str list) lists directories in which to look for our .bil and .hdr files, as well as metadata in .mat files
    udlr_dir str directory in which to search for udlr renders of the surface normal of the 3D model
    force = True # should we overwrite existing pickled datafiles?
    mainplots = False # generate the 'main' figures for the analysis?
    additionalplots = False # generate additional figures for the analysis? 
    label_dict, color_dict are dictionaries that convert species acronyms (H_, CR_, CM_) into text labels and colors
    color_scale (float) is a multiplier for the brightness of the false sRGB colors generated in the analysis
    """
    
    np.random.seed(10) # set random seed for reproducibility

    directories = ["../dat/Normal_hyperspectral_images", 
                   "../dat/Tail_hyperspectral_images", 
                   "../dat/Tilted_hyperspectral_images"]    
    
    for directory in directories: 
        print('Working on ' + directory)
        # to allow flexibility, we list the location of datafiles in a text file
        datlookuptable = directory + ".txt"

        # find the datafiles
        datafiles = LookUpData(open(datlookuptable, "r"), lookin) 
        utils.CheckFileMatch(datafiles["matfiles"], datafiles["hdrfiles"], datafiles["bilfiles"]) # check each list is in the same order

        # load an associated mat file, list the available masks
        image_example = 0
        mat_contents = sio.loadmat(datafiles["matfiles"][image_example], struct_as_record=False, squeeze_me=True)
        mat_contents = mat_contents['ImStruct']

        # Find metadata about mask availability for each file (i.e. what masks do we have in what image?)
        matinfo = bop_sampling.FindMasks(datafiles, directory)

        # Find the samples for each - spread desired samples across the images per specimen in which a patch is visible
        desire = 1000 # how many spectra would we want per bird per patch? 
        desirematrix = desire/matinfo['divider'] # should be desired sample given the number of facings for each bird.
        desirematrix[matinfo['hasmask']==0]=0 # missing masks shouldn't matter but setting to 0 anyway
        print('Expect divide by zero error, this is just images in which patches do not occur')

        if os.path.isfile(directory + '/Gatherdata_mask_0.pickle') and force == False: # Note: this checking will fail to work if some but not all present
            print('Pickled data already exist for ' + directory)
        else:
            # Generate & gather data
            if force == True:
                print('Pickled data already exist for ' + directory)
                print('But force is on in RunDataAnalysis, so running!')

            t = time.time()

            # do sampling
            for k in np.arange(len(mat_contents.bwlist))[np.sum(desirematrix, axis = 0)>0]:
                ProcessData(datafiles, udlr_dir, k, desirematrix, directory, color_scale = color_scale)

            elapsed = time.time() - t

            print(directory + ' done in ' + str(elapsed/60) + ' minutes') # how long did it all take?
                
            
    if mainplots: # Now automate generation of the figures for the paper here
        
        t = time.time()
        
        # dSh umap
        pltn = 'dSh_umap_embedded' # can choose any of tSNE, umap, cmds measures of KLPD or dSh
        fig = bop_plotting.BasicPlot(pltn, mat_contents, label_dict, color_dict, sort_by_rgb = False)
        fig.savefig('../out/Figures/' + pltn + '.png', dpi = 300, bbox_inches="tight")
        fig.savefig('../out/Figures/' + pltn + '.pdf', dpi = 300, bbox_inches="tight")

        # KLPD umap
        pltn = 'KLPD_umap_embedded' # can choose any of tSNE, umap, cmds measures of KLPD or dSh
        fig = bop_plotting.BasicPlot(pltn, mat_contents, label_dict, color_dict, sort_by_rgb = False)
        fig.savefig('../out/Figures/' + pltn + '.png', dpi = 300, bbox_inches="tight")
        fig.savefig('../out/Figures/' + pltn + '.pdf', dpi = 300, bbox_inches="tight")

        # Basic plots for Lab
        pltn = 'Lab' # 'Lab' gives Lxa, others clearer
        fig = bop_plotting.BasicPlot(pltn, mat_contents, label_dict, color_dict, lab = 'avb', sort_by_rgb = True)
        fig.savefig('../out/Figures/CIELAB.png', dpi = 300, bbox_inches="tight")
        fig.savefig('../out/Figures/CIELAB.pdf', dpi = 300, bbox_inches="tight")

        # CieXYy plots
        includeWhole = 1 # include the first mask in the columns or exclude - as defined, this is whole mask
        fig = bop_plotting.CIEXYyPlot_two(mat_contents, includeWhole, label_dict, color_dict, sort_by_rgb = True)
        fig.savefig('../out/Figures/CIEXYZ.png', dpi = 300, bbox_inches="tight")
        fig.savefig('../out/Figures/CIEXYZ.pdf', dpi = 300, bbox_inches="tight")

        # 2D Tetracolor plot
        # %matplotlib inline
        fig = bop_plotting.tXYZplot(mat_contents, 1,  label_dict, color_dict, 'u', sort_by_rgb = True)
        fig.savefig('../out/Figures/tXYZ.png', dpi = 300, bbox_inches="tight")
        fig.savefig('../out/Figures/tXYZ.pdf', dpi = 300, bbox_inches="tight")

        # First 2 pca axes
        pltn = 'pcasam_out' # this is PCA where the mean has been subtracted from all the samples
        fig = bop_plotting.BasicPlot(pltn, mat_contents, label_dict, color_dict, sort_by_rgb = False)
        fig.savefig('../out/Figures/' + pltn + '.png', dpi = 300, bbox_inches="tight")
        fig.savefig('../out/Figures/' + pltn + '.pdf', dpi = 300, bbox_inches="tight")

        # PCA umap embedding
        pltn = 'pca_umap_embedded' # this is PCA where the mean has been subtracted from all the samples
        fig = bop_plotting.BasicPlot(pltn, mat_contents, label_dict, color_dict, sort_by_rgb = False)
        fig.savefig('../out/Figures/' + pltn + '.png', dpi = 300, bbox_inches="tight")
        fig.savefig('../out/Figures/' + pltn + '.pdf', dpi = 300, bbox_inches="tight")

        # First 2 pcas without subtracting brightness
        pltn = 'pcasam_out_bright' 
        fig = bop_plotting.BasicPlot(pltn, mat_contents, label_dict, color_dict, sort_by_rgb = False)
        fig.savefig('../out/Figures/' + pltn + '.png', dpi = 300, bbox_inches="tight")
        fig.savefig('../out/Figures/' + pltn + '.pdf', dpi = 300, bbox_inches="tight")
        
        # PCA umap, without subtracting brightness
        cln = 'species'
        pltn = 'pca_umap_embedded_bright'
        fig = bop_plotting.BasicPlot(pltn, mat_contents, label_dict, color_dict, sort_by_rgb = False)
        fig.savefig('../out/Figures/' + pltn + '.png', dpi = 300, bbox_inches="tight")
        fig.savefig('../out/Figures/' + pltn + '.pdf', dpi = 300, bbox_inches="tight")
        
        elapsed = time.time() - t

        print('Main plots done in ' + str(elapsed/60) + ' minutes') # 

    if additionalplots:
        
        t = time.time()

        # RNL colorspace plot
        fig = bop_plotting.RNL_multi(label_dict, color_dict)

        # 3D tetraplot
        fig = bop_plotting.tcs_multi(label_dict, color_dict)

        # plot out the mask identifications for all images
        # MaskPlotting()

        # plot all samples for the angled ventral view
        bop_plotting.SampleCheck_breast(lookin, color_dict = color_dict) 

        # plot all samples for all normal views, since this involves all images this is very slow and memory intensive
        # bop_plotting.SampleCheck_half(pas = [0, 2, 3, 4, 6], halfSpecimens = False, lookin = lookin, color_dict = color_dict) # will run on all patches
        
        # this plots the whole sample, but for only half the specimens - for main figure
        bop_plotting.SampleCheck_half(pas = range(1), halfSpecimens = True, lookin = lookin, color_dict = color_dict)

        # here re-calculate the UMAP embedding for SI plots using different n_neighbors and min_dist for umap
        pltn = 'None' # since reembedding, using only PCA (hardcoded)
        fig = bop_plotting.BasicPlot(pltn, mat_contents, label_dict, color_dict, reembed = True, n_neighbors = 2000, min_dist = 0.1, sort_by_rgb = False)
        # fig.suptitle('n_neighbors:' + str(2000) + ', min_dist:' + str(0.1), fontsize=16)
        fig.savefig('../out/Figures/pcaUMAP_params_nn2000_md01.png', dpi = 300, bbox_inches="tight")
        fig.savefig('../out/Figures/pcaUMAP_params_nn2000_md01.pdf', dpi = 300, bbox_inches="tight")
        
        pltn = 'None' # since reembedding, using only PCA (hardcoded)
        fig = bop_plotting.BasicPlot(pltn, mat_contents, label_dict, color_dict, reembed = True, n_neighbors = 15, min_dist = 1, sort_by_rgb = False)
        # fig.suptitle('n_neighbors:' + str(15) + ', min_dist:' + str(1), fontsize=16)
        fig.savefig('../out/Figures/pcaUMAP_params_nn15_md1.png', dpi = 300, bbox_inches="tight")
        fig.savefig('../out/Figures/pcaUMAP_params_nn15_md1.pdf', dpi = 300, bbox_inches="tight")
        
        elapsed = time.time() - t

        print('Additional plots done in ' + str(elapsed/60) + ' minutes') # 


def ProcessData(datafiles, udlr_dir, k, desirematrix, directory, window_length = 45, color_scale = 1, saveout = True):
    """
    Code that generates samples for a given mask k (that is plumage patch) across all the hyperspectral images 
    listed in datafiles, taking approx the number of samples from desirematrix, and pickling the output to directory.
    
    Argument datafiles is an ordered dict of the 'hdrfiles' (.hdr for each .bil), 
    the 'bilfiles' (each hyperspectral image), and 'matfiles' (each set of metadata generated in 
    matlab, saved in .mat format). k indicates the index of which mask from matfiles we're interested in
    sampling across the hyperspectral images.
    desirematrix is the pre-generated matrix which indicates how many samples we want from a given
    patch/image/specimen combination. directory indicates where to save .pickle files of the processed data.
    window length (integer) controls the degree of filtering/smoothing we apply to the spectra. Our main analysis uses 45, we
    expose this parameter only in order to make one supplemental plot omitting this smoothing.
    color_scale (float) is a multiplier for the brightness of the false sRGB colors generated in the analysis
    saveout (boolean) is whether to save the picked output dat in directory (otherwise only return it)
    """

    # sample the hyperspectral image k
    sample_extent = 4
    arr, meta, owl = bop_sampling.SampleMask(datafiles, udlr_dir, desirematrix, sample_extent, k, plotme = 'false') 
    
    # interpolate and extrapolate sample 300-700, in 1nm intervals, with some filtering
    wl, arr = utils.InterpExtrapSample(owl, arr, np.arange(325, 701, 1),  wlout = np.arange(300, 701, 1), window_length = window_length)
    
    # get color info
    XYZ, sRGB, Lab = utils.GetColors(arr, wl, color_scale)
    usml, relusml, txyz = utils.Spec2usml(arr, wl)
    
    # calc KLPD and dSh
    KLPD, dSh, dW = utils.GetKLPD(arr)
    
    # calc umap embeddings of KLPD and dSh
    dSh_umap_embedded = utils.EmbedDist(dSh)
    KLPD_umap_embedded = utils.EmbedDist(KLPD)
    
    # also get PCA coordinates, along with umap embeddings of all PCA axes
    pcasam_out = utils.EmbedPCA(arr, subtract_mean = True) # with mean subtracted
    pca_umap_embedded = utils.EmbedDist(None, pcasam_out)
    
    pcasam_out_bright = utils.EmbedPCA(arr, subtract_mean = False) # withough mean subtracted
    pca_umap_embedded_bright = utils.EmbedDist(None, pcasam_out_bright)

    # save out result
    dat = GatherProcess(wl, arr, meta,
            XYZ, sRGB, Lab,
            usml, relusml, txyz,
            dSh_umap_embedded,
            KLPD_umap_embedded,
            pcasam_out, pca_umap_embedded,
            pcasam_out_bright, pca_umap_embedded_bright)
    
    if saveout == True:
        SaveGather(k, dat, directory)
    
    return dat
    
def GatherProcess(wl, arr, meta,
            XYZ, sRGB, Lab,
            usml, relusml, txyz,
            dSh_umap_embedded,
            KLPD_umap_embedded, 
            pcasam_out, pca_umap_embedded,
            pcasam_out_bright, pca_umap_embedded_bright):

    """ Utility function to make dict from processed data """
    
    # make dict for convenience
    vars = ['wl', 'arr', 'meta',
            'XYZ', 'sRGB', 'Lab',
            'usml', 'relusml', 'txyz',
            'dSh_umap_embedded',
            'KLPD_umap_embedded',
            'pcasam_out', 'pca_umap_embedded',
            'pcasam_out_bright', 'pca_umap_embedded_bright']
    dat = {}
    for variable in vars:
        dat[variable] = eval(variable)
    return dat
    
def LoadGather(k, directory = None):

    """ Utility function to load saved gathered processed data """
    
    if directory is None:
        error('Give load gather a directory')
    else:
        fn = directory + r'/Gatherdata_mask_' + str(k) + '.pickle'
    # Load previously saved set of measurements
    dat = utils.pick(fn, ls = 'load')    
    return dat


def SaveGather(k, dat, directory = None):

    """  Utility function to save gathered processed data """
    
    if directory is None:
        error('Give save gather a directory')
    else:
        fn = directory + r'/Gatherdata_mask_' + str(k) + '.pickle'
    # save
    utils.pick(fn, what = dat, ls = 'save')    


def LookUpData(lines, lookin):

    """ Utility function to look across a few folders, and find where the .mat, .bil, and .hdr files for a given list of basenames can be found """
    
    lines = lines.read().split('\n')
    matfiles = []
    hdrfiles = []
    bilfiles = []
    
    for l in lines:
        for k in lookin:
            if os.path.exists(k+l):
                if ".mat" in k+l:
                    matfiles.append(k+l)
                elif ".bil.hdr" in k+l:
                    hdrfiles.append(k+l)
                else:
                    bilfiles.append(k+l)

    out =  {'hdrfiles': hdrfiles, 'bilfiles':bilfiles, 'matfiles':matfiles}
    return out

