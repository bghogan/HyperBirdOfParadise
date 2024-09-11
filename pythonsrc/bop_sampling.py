## Functions specific to the BOP data-set 

## SAMPLING ##

import os
import re
import numpy as np
from scipy import io as sio
import spectral.io.envi as envi

import hsi.utils as utils
import bop_analysis
import bop_udlr

def FindMasks(datafiles, saveout):

    """ Loop through each hsi and its metadata, look through each patch/mask, and note down which images have which patches/masks.
    Also work out a divider for each image/patch such that samples for patch 1 on specimen 1 will be split evenly between all HSIs of
    specimen 1 that contain patch 1 (in a mask with more than 1000 pixels). Pickle the results out. """
    
    # initial load, so we can ascertain the number of masks over which to loop
    mat_contents = sio.loadmat(datafiles["matfiles"][0], struct_as_record=False, squeeze_me=True)
    mat_contents = mat_contents['ImStruct']

    # generate file name to save out mask information to
    # this will basically say for which specimen and patch and image, how many samples should we try to generate
    matfilename = saveout + '/matinfo' + '.pickle'

    if os.path.isfile(matfilename):
        matinfo = utils.pick(matfilename, ls = 'load')
        print('Previous matfile loaded')
    else:

        # make regex pattern for extracting individual id from filenames
        pattern = re.compile(r".*(?=_.*$)") # this gets the specimen
        pattern2 = re.compile(r"(?<=_)[A-z]*$") # this gets the facing

        # init some information
        hasmask = np.zeros((len(datafiles["matfiles"]), mat_contents.masklist.shape[0]))
        hitmask = np.zeros((len(datafiles["matfiles"]), mat_contents.masklist.shape[0]))
        names = []
        facing = []

        # init a matrix to save out the number of samples we'll collect
        maskid = np.tile(np.arange(0,mat_contents.masklist.shape[0],1), [len(datafiles["matfiles"]), 1])

        # for each of hyperspectral images, open each of the masks and generate some sampling based on desire
        for i, j in zip(range(len(datafiles["matfiles"])), datafiles["matfiles"]):
            # print('Working on ' + j)
            print(str(i+1)+' of '+str(len(datafiles["matfiles"])))
            # extract individual from file name
            res = re.findall(pattern, os.path.basename(os.path.splitext(j)[0]))[0]
            names.append(res)
            # extract facing from file name
            res = re.findall(pattern2, os.path.basename(os.path.splitext(j)[0]))
            facing.append(res)

            # Import matlab generated image metadata
            mat_contents = sio.loadmat(j, struct_as_record=False, squeeze_me=True)
            mat_contents = mat_contents['ImStruct']
            for k in range(mat_contents.masklist.shape[0]): # loop through masks 
                # get mask
                mask = mat_contents.masklist[k]
                # if exists, mask it down
                if not np.isnan(mask).all():
                    # mark whether mask available (1 or 0), and note how many pixels are in the mask
                    hasmask[i, k] = 1
                    hitmask[i, k] = mask.sum()

        # exclude any tiny masks which bias the sampling toward them
        hasmask[hitmask<=1000]=0 

        # make matrix of individual
        names = np.array(names)
        un, group = np.unique(names, return_inverse = 'true')

        # get matrices for mask identity and individual identity
        individual = np.transpose(np.tile(group, [mat_contents.masklist.shape[0],1]))

        # find how many masks per patch per bird.
        divider=np.zeros((len(datafiles["matfiles"]), mat_contents.masklist.shape[0]))
        for indi in range(len(un)):
            for ma in range(mat_contents.masklist.shape[0]):
                divider[(individual==indi) & (maskid == ma)]=sum(hasmask[(individual==indi) & (maskid == ma)])

        # save this information out in a dict
        matinfo = {'divider': divider, 'names': names, 'maskid': maskid, 'hasmask': hasmask, 'hitmask': hitmask, 'individual': individual, 'facing':facing}

        # pickle that
        utils.pick(matfilename, matinfo, ls = 'save')
        
    return matinfo

    print('Done finding mask presence')

# Loop through birds, collect spectra from the masks in order to try to get to desired n samples per bird
def SampleMask(datafiles, udlr_dir, desirematrix, n, k, plotme = 'false'):

    """ Look through the HSI images (datafiles) for patch number k, using a sampling extent n (sample is mean of x-n to x+n, y-n to y+n), and using a matrix 
    (n HSI images x n masks/patches) with a desired number of samples to collect. Collect all the samples from the images, return sample spectra, a dict
    of metadata, and the wavelengths for the spectra. udlr_dir indicates where to look for the renders of the surface normal for 3D models of the hyperspectral 
    images"""
    
    # say what we're doing
    print('Sampling across all images for mask ' + str(k))
    
    # init empty output array (dim 2 unknown at start)
    img = envi.open(datafiles["hdrfiles"][0], datafiles["bilfiles"][0])
    arr = np.empty((0, len(img.bands.centers)))
    
    # init other outputs
    listmask = np.empty((0, 0))
    listspecimen = np.empty((0, 0))
    listspecies = np.empty((0, 0))
    listfacing = np.empty((0, 0))
    listlocationx = np.empty((0, 0))
    listlocationy = np.empty((0, 0))
    listn = np.empty((0, 0))
    
    # keep path too, that way we know where it all came from easily
    listpath = np.empty((0, 0))
    
    # append also the lr and up values from normal maps
    listlr = np.empty((0, 0))
    listud = np.empty((0, 0))
    listde = np.empty((0, 0))

    # loop through all mat/bil/hdr files
    for i, j in zip(range(len(datafiles["matfiles"])), datafiles["matfiles"]):   
        
        # print('Working on ' + j)
        print(str(i+1)+' of '+str(len(datafiles["matfiles"])))

        # Import matlab generated image metadata
        mat_contents = sio.loadmat(j, struct_as_record=False, squeeze_me=True)
        mat_contents = mat_contents['ImStruct']

        # get mask up
        mask = mat_contents.masklist[k]
        
        # import udlr
        
        flr = udlr_dir + utils.GetFilename(datafiles["bilfiles"][i]) + '_normal_R.tif'
        if os.path.isfile(flr):
            lr, ud, ma, de = bop_udlr.GetUDLR(udlr_dir, datafiles["bilfiles"][i])

        # if mask exists load and sample
        if not np.isnan(mask).all():

            # find correct desired sampled spectra
            desire = desirematrix[i,k]

            # find the locations for samples
            xs, ys = utils.SampleSelection(mask, desire, n, plotme = plotme, res = 200)

            # load samples in, append to output
            img = envi.open(datafiles["hdrfiles"][i], datafiles["bilfiles"][i])
            Out = utils.ReadBands(img, xs, ys, n)
            Out = Out / mat_contents.ref99[None,:] # use reference measure from metadata to correct to reflectance
            arr = np.append(arr, Out, axis = 0)
            # also pass owl
            owl = img.bands.centers
            
            # We'll want to collect some extra information about the samples contained in arr
            if os.path.isfile(flr): # Look up the udlr images
                print('Finding udlr for samples')
                for xsloop in range(len(xs)): # for each sample in turn

                    # find range so we can accurately average angles
                    xmi = xs[xsloop]
                    xma = xs[xsloop]+n
                    ymi = ys[xsloop]
                    yma = ys[xsloop]+n

                    # find the mean of each of our items (udlr) for the samples
                    mud = np.nanmean(ud[ymi.astype(int):yma.astype(int), xmi.astype(int):xma.astype(int)])
                    mlr = np.nanmean(lr[ymi.astype(int):yma.astype(int), xmi.astype(int):xma.astype(int)])
                    mde = np.nanmean(de[ymi.astype(int):yma.astype(int), xmi.astype(int):xma.astype(int)])                

                    # append udlr and depth to the lists to save out
                    listud = np.append(listud, mud)
                    listlr = np.append(listlr, mlr)
                    listde = np.append(listlr, mde)
    
            # Mask, Specimen, Species, facing, locations
            listmask = np.append(listmask, np.repeat(k, Out.shape[0]))
            listspecimen = np.append(listspecimen, np.repeat(utils.GetSpecimen(j), Out.shape[0]))
            listspecies = np.append(listspecies, np.repeat(utils.GetSpecies(j), Out.shape[0]))
            listfacing = np.append(listfacing, np.repeat(utils.GetFacing(j), Out.shape[0]))
            listlocationx = np.append(listlocationx, xs)
            listlocationy = np.append(listlocationy, ys)
            listn = np.append(listn, np.repeat(n, Out.shape[0]))
            
            # path too
            listpath = np.append(listpath, np.repeat(j, Out.shape[0]))

    # make dict of these metadata
    meta = {'mask': listmask, 
            'specimen': listspecimen, 
            'species': listspecies, 
            'facing': listfacing, 
            'path': listpath,
            'x':listlocationx, 
            'y':listlocationy, 
            'n':listn,
            'listlr':listlr,
            'listud':listud,
            'listde': listde}
        
    # plot all those spectra
    if plotme == 'true':
        utils.PlotSpec(img, x, y, Specs = arr)
    
    return arr, meta, owl
    