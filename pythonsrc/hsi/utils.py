################ This file contains HSI functions ################

# Used in a several places
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import colour
import re

## Used only in some functions
import pandas as pandas # used in Spec2usml
import math # used in Spec2usml
from skimage.morphology import binary_erosion as erode # used in SampleSelection
from scipy.signal import savgol_filter # used in InterpExtrapSample
from scipy.interpolate import PchipInterpolator # used in InterpSample and InterpExtrapSample
from scipy.interpolate import interp1d # used in InterpSample and InterpExtrapSample
from matplotlib.colors import Normalize # used in PlotFalseRGB
import time # only used to time umap embedding in EmbedDist
import umap.umap_ as umap  # only used to time umap embedding in EmbedDist
from sklearn.decomposition import PCA # only used in EmbedPCA
from colour.models import RGB_COLOURSPACE_sRGB # used in getcolors

################ Generic helpers ################

def pick(filename, what = 'NaN', ls = 'save'):
    """ Utility function just to tidy up pickling, 
    
    Arguments:
    filename - string directory and file name to save to or load
    what - optional object to pickle
    ls - 'load' or 'save', what action to take """
    
    if ls == 'save': # saving
        outfile = open(filename,'wb')
        pickle.dump(what,outfile)
        outfile.close()
    else: # loading
        infile = open(filename,'rb')
        what = pickle.load(infile)
        infile.close()
        return what

################ Functions relating to reading spectral bands/spectra ################

def ReadBands(img, xs, ys, n = 0):
    """
    Function to read spectra from a hyperspectral image img (generated with spectral.io.envi.open), at location a location or list of locations
    
    Arguments:
    img - hyperspectral image img (generated with img = spectral.io.envi.open)
    xs, ys - numpy arrays of sample x and y coordinates
    n - optional sampling extent, if not zero then return the mean of a subregion defined by x-n:x+n, y-n:y+n
    
    Returns sampled spectra in a sample x wavelength numpy array
    
    """
     
    if n>0: # if 
        print('Samples need averaging')
        Specs = ReadAverageBands(img, xs, ys, n)

    else:
        print('Samples do not need averaging, processing')
        Specs = ReadSingleBands(img, xs, ys)
        

    return Specs

def ReadSingleBands(img, x, y):
    """ Read and flatten multiple sample locations at once from hyperspectral image 'img' (generated with spectral.io.envi.open),
    at locations x and y (np.arrays of integer sample locations of equal length)
    """

    # Init empty specs
    Specs = np.empty((len(x), len(img.bands.centers)))
    Specs[:] = np.NaN

    # Load each spectrum in a loop
    for i in range(Specs.shape[0]):
        if (i % 500 == 0) & len(x)>1000: # occasionally print where we are in the loop
            print('Loading pixel ' + str(i) + ' of ' + str(len(x)))
        Specs[i,:] = img[x[i], y[i], :].flatten()
    return(Specs)
            
def ReadAverageBands(img, xs, ys, n):
    """ Read and flatten multiple sample locations at once from hyperspectral image 'img' (generated with spectral.io.envi.open),
    at locations x and y (np.arrays of integer sample locations of equal length)
    in this case, we extend our samples spatially by n pixels, so xs:xs+n and ys:ys+n are read, and averaged """

    if len(xs) != len(ys):
        raise Exception("Length of x, y vector of samples must be equal length")
    
    # Init empty specs
    Specs = np.empty((len(xs), len(img.bands.centers)))
    Specs[:] = np.NaN
    
    print('Processing and averaging')

    # Load each sample in loop, and average
    for loop in range(Specs.shape[0]): # each set to average, i.e. num samples, inefficient
        if (loop % 500 == 0) & len(xs)>1000: # occasionally print where we are in the loop
            print('Averaging '+ str(loop) + ' of ' + str(len(xs)))
        # use sub-image loading
        xmi = xs[loop]
        xma = xs[loop]+n
        ymi = ys[loop]
        yma = ys[loop]+n
        thisloop = img.read_subregion(col_bounds = [xmi, xma], row_bounds = [ymi, yma], use_memmap=True)
        thisloop = np.mean(thisloop, axis = (0,1))
        Specs[loop, :] = thisloop
    return(Specs)

def PlotSpec(img = float("nan"), x= float("nan"), y= float("nan"), Specs = float("nan"), wl = float("nan"), ylabel = "Reflectance (0-1)"):
    """ Small function to plot a, or set of spectra from the native hyperspectral image (generated with spectral.io.envi.open)
    can take either img and x and y locations OR an np.array of some spectra to plot, along with the wavelength bands
    """
    
    if np.isnan(Specs).all():
        Specs = ReadBands(img, x, y)
    # if we don't have a vector of the wavelength bands for specs, load them from the image
    if np.isnan(wl).all():
        wl = img.bands.centers
    # plot
    plt.plot(wl, Specs.T)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(ylabel)
    plt.show()

################ Functions that interpolate and extrapolate spectra ################

def InterpSample(wl, arr, wlnew = np.arange(325, 700, 1)):
    """ 
    Crop zeros, interpolate spectra to bands in a new defined range
    
    Arguments:
    wl - numpy array (n wavelength) of band centers of original array 
    arr - numpy array (sample x wavelength) of spectra to interpolate 
    wlnew: numpy array of band centers desired, default is 325-700 nm at 1nm intervals
    
    Note: interpolator does not have filtering stage, see InterpExtrapsample() for filtering
    """
    
    # Gen pchip interpolator
    intfun = PchipInterpolator(x = wl, y = arr, axis = 1, extrapolate=True)
    # Interp sample
    arr = intfun(wlnew)  
    arr[arr<=0]=0.00001 # below zero to very small value
    wl = wlnew

    return wl, arr


def InterpExtrapSample(wl, arr, wlnew = np.arange(325, 701, 1), wlout = np.arange(300, 701, 1), window_length = 10):
    """ 
    
    Cut a spectrum to the lower of a linear set of wavelengths (wlnew), then filter/smooth the spectra. Then interpolate the result to 1nm bins, using pchip.
    Then extrapolate from wlnew.min() to a new set of wavelengths, using pchip extrapolation
    
    Arguments:
    wl - numpy array of original band centers for array arr
    arr - np.array of spectra (sample x wavelength)
    wlnew - np.array of wavelengths to interpolate to, with the lower number being a cut-off below which is extrapolated 
    wlout - np.array of wavelengths to extrapolate/interpolate to, extrapolation by default is 325 to 300 (the area of non-overlap with wlnew).
    window_length - controls the window over which signal filtering/smoothing occurs.
    
    """

    # cut array and wl at min of wlnew
    print('Cutting original spectra to >' + str(wlnew.min()))
    owl = np.array(wl)
    carr = arr[:,owl>wlnew.min()]
    cwl = owl[owl>wlnew.min()]

    # interpolate to 1nm intervals, filter 
    if window_length == 0:
        print('No savgol_filter applied, extrapolation will just use last value')
        filtered = carr
        inter = PchipInterpolator(x = cwl, y = filtered, axis = 1, extrapolate=False) # pchip to 1nm, but no extrapolation
        inter = inter(wlnew[1:]) # interpolate wavelength values to 326-700
        extra = interp1d(x = wlnew[1:], y = inter, axis = 1, fill_value=(filtered[:, 0], filtered[:, -1]), bounds_error=False) # extrapolation with last values
        extra = extra(wlout)
    else:
        print('Filtering spectra with savgol_filter, extrapolation will use pchip')
        filtered = savgol_filter(carr, window_length = window_length, polyorder = 2, axis=1, mode='interp', cval=0.0)
        extra = PchipInterpolator(x = cwl, y = filtered, axis = 1, extrapolate=True)
        extra = extra(wlout)

    print('Interpolated spectra to 1nm from ' + str(wlnew.min()) + ' to ' + str(wlnew.max()))

    print('Spectra now ' + str(wlout.min()) + ' to ' + str(wlout.max()))
    extra[extra<=0]=0.00001 # below zero and zero reflectances changes to very low constant (useful for some computations)
    
    return wlout, extra

################ Helpers for sampling ################

def SamplesGivenGap(mask, desire = 2000):
    """ 
    Proposes a gap (in pixels) between samples, given desired number samples and a binary mask image
    assumes a square grid of samples, but should also approximate any rotation of same grid. Probably not
    a hexagonal or other grid with different packing though.
    
    Arguments:
    mask - np.array (mask image) of 1s and 0s
    desire - integer, number of samples to attempt to gather
    
    """
    
    # what range of gap should we consider?
    ranges = np.arange(1, 100, 1) # max gap is 100 pixels, very dependent
    
    # how many samples given gap ranges, rounded
    guess = mask.sum() / ranges**2
    guess = guess.round()
    
    # find the closest gap given desired number of samples
    result = guess - desire
    pick = abs(result).min()
    pick = np.where(abs(result)==pick) # there can be more than one best gap, if evenly distance from the ideal number
    
    # but we only want to return one likely gap length
    gap = ranges[pick]
    gap = gap[0] # if more than one closest, pick wider gap
    
    return(gap)

def SampleSelection(mask, desire = 1000, n = 5, plotme = True, res = 200):
    """
    This code returns the location and optionally plots samples given a binary mask image and desired number of samples

    Arguments:
    mask - np.array (mask image) of 1s and 0s
    desire - integer value of number of samples wanted
    n - sample extent to indicate on the plot
    plotme - boolean whether to plot
    res - resolution at which to plot
    
    Returns xs and ys, lists of sample locations (for use in ReadBands or similar)
    
    """
    
    # erode mask by n to disallow samples falling exactly on mask boundary (so that some pixels would
    # come from outside of the mask, if any spatial averaging used).
    for k in range(n):
        if k == 0:
            emask = mask
        emask = erode(emask)

    # estimate gap between samples given desired n samples and mask
    gap = SamplesGivenGap(emask, desire)

    # find where mask starts and stops in x and y
    x = np.where(emask.sum(axis=0)>=1) # get 1's as columns
    y = np.where(emask.sum(axis=1)>=1) # get 1's as rows
    x = [np.min(x), np.max(x)]
    y = [np.min(y), np.max(y)]

    # make a grid from these coords
    xc = np.arange(x[0], x[1], gap)
    yc = np.arange(y[0], y[1], gap)
    xx, yy = np.meshgrid(xc, yc)
    
    # only sample where emask = 1 (mask may not be concave)
    xs = xx[emask[yy,xx]==1] 
    ys = yy[emask[yy,xx]==1]

    # how many samples do we get at this level of gap?
    print('Selected ' + str(xs.shape[0]) + ' samples') 
    
    # plot the outcome
    if plotme==True:

        # find locations for sampling subimages to spatially average
        xmi = xs
        xma = xs+n
        ymi = ys
        yma = ys+n
        
        # gen mask for plotting (including erosion)
        pmask = mask
        pmask[emask==1]=2
        pmask = np.round(pmask)
        
        # make plot, show image
        fig, ax = plt.subplots(1, figsize=(12,8), dpi=res, facecolor='w', edgecolor='k')
        ax.imshow(pmask, cmap = 'Set1')
        # fig.colorbar(im)
        
        # scatter
        ax.scatter(x=xx, y=yy, c='r', s=1) # plot sampling grid
        ax.scatter(x=xs, y=ys, c='g', s=1) # plot center samples
        
        # limit to mask
        ax.set_xlim(x[0], x[1])
        ax.set_ylim(y[1], y[0])
        
        # if sampling squares, plot each
        if n>0:
            import matplotlib.patches as patches
            for loop in range(len(xmi)):
                # generate rectangles
                rect = patches.Rectangle((xmi[loop],ymi[loop]),n,n,linewidth=0.5,edgecolor='r',facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)
                
        plt.show()
    
    return xs, ys
    
################ Function for KLPD distance ################
    
def GetKLPD(sel):

    
    from tqdm import tqdm # adds a progress bar, and eta for calculation
    
    """
    Function calculates KLPD distance on one sample (all pairwise differences), as well as the shape dSh and brightness dW components thereof
    This is optimized for generating all pairwise distances of the spectra
    Note that we are simplifying the calculation by integrating by summation, rather than using trapezoidal approximation
    This is a memory hungry approach
    
    Arguments:
    sel - np.array of a selection of spectra 'sel' (samples x wavelengths numpy array) 

    Returns KLPD, dSh, dW distance matrices (samples x samples)
    """
    
    # if any spectral readings below 0, break
    if(np.any(sel<=0)):
        raise Exception("KLPD needs all >0 values")
    
    # get brightness of each spectrum
    k = np.sum(sel, 1) # eq 18/19
    
    # do logging and normalizing outside of loop for speed
    S = (sel.T/k).T # normalize spectra to sum to 1, eq 19
    lS = np.log(S)
    lk = np.log(k)
    
    # preallocate output
    KL12 = np.empty((sel.shape[0], sel.shape[0]))
        
    # do calculation in loop
    for row in tqdm(range(sel.shape[0])):
        
        # get KL12 
        KL12[row,:] = np.sum(np.multiply(S[row], np.subtract(lS[row], lS)), 1) # eq 20, note log(x)-log(y) = log(x/y)
        # np.sum(S[5] * np.log(S[5]/ S[10])) == KL12[5,10] # a check on this calculation and the placement in matrix
    
    # generate KL21
    KL21 = KL12.T # because all pairwise and designed to be symmetric (a v b = b v a), inverse is just transposed original
    
    # generate dSh
    dSh1 = np.multiply(k, KL12) # eq 24 (top part for dSh)
    dSh2 = dSh1.T # same transposition here
    dSh =  dSh1 + dSh2 # eq 23
    
    # generate dW 
    dW = np.multiply(np.subtract(k, k[:,None]), np.subtract(lk[None,:], lk[:,None])) # eq 24, bottom dW part
    # dW[5, 10] == np.subtract(k[5], k[10]) * np.subtract(lk[5], lk[10]) # a check on this calculation and the placement in matrix
    
    KLPD = dSh + dW
    
    return KLPD, dSh, dW

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """
    Utility function to check that a matrix is symmetric (within relative and absolute tolerances, see numpy.allclose())
    see https://stackoverflow.com/questions/42908334/checking-if-a-matrix-is-symmetric-in-numpy
    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

################ Functions for umap and pca embedding ################

def EmbedDist(DistMatrix = None, Coordinates = None, n_neighbors = 15, min_dist = 0.1, verbose = 1):
    """ Convenience function to calculate umap embeddings of distance matrices or coordinates,
    Can take either a precomputed distance matrix, or a dataset of coordinates,
    Also takes two main parameters of umap embedding, n_neighbors and min_dist. 
    See UMAP documentation for more information. """

    if DistMatrix is None:
        if Coordinates is None:
            raise Exception('Need either distance matrix or coordinates to embed with umap')

    if DistMatrix is not None:
        if Coordinates is not None:
            raise Exception('Both coordinates and distance matrix passed for umap embed, pick one please')

    t = time.time()
    if Coordinates is None:
        print('Computing umap on distance matrix')
        umap_embedded = umap.UMAP(metric='precomputed', verbose = verbose, n_neighbors = n_neighbors, min_dist = min_dist).fit_transform(DistMatrix)
    else:
        print('Computing umap on coordinates')
        umap_embedded = umap.UMAP(metric='euclidean', verbose = verbose, n_neighbors = n_neighbors, min_dist = min_dist).fit_transform(Coordinates)

    elapsed = time.time() - t
    print('UMAP done in ' + str(elapsed))    

    return umap_embedded


def EmbedPCA(Spec, subtract_mean = False, n_comp = 20):
    """ Convenience function to calculate PCA on given spectra (sample x wavelength np.array), 
    # including an option to subtract the mean from each spectrum,
    # and the number of PCA components to compute. """
    
    # subtract the mean from each spectrum to remove brightness
    if subtract_mean:
        pcasam = np.subtract(Spec.T, np.mean(Spec, axis = 1)).T # could try division too
    else:
        pcasam = Spec

    # take the minimum of the number of spectra and n_comp components
    n_comp = np.min([n_comp, Spec.shape[0]]) 
    
    # compute PCA
    pca = PCA(n_components = n_comp) # set the number of components 
    pca.fit(pcasam, ) # fit the pca
    pcasam_out = pca.transform(pcasam) # make a transformed dataset
    
    return pcasam_out

################ Helpers for filename/file checking ################

def GetFilename(path):
    """ Get rid of path and extension of filename, allowing for filenames with double extensions such as xxx.ext.ext """
    base = os.path.basename(path)
    name = os.path.splitext(base)
    name = name[0]
    name = os.path.splitext(name) # since some have .bil.hdr, must do twice
    name = name[0]
    return(name)

def GetFacing(path):
    """ From filepath, get the facing (assuming format is like CM_M_01_Back.bil: species_sex_specimen_facing) """
    na = GetFilename(path)
    pattern = re.compile(r"(?<=_)[A-z]*$") # this gets the facing
    res = re.findall(pattern, na)
    return res

def GetSpecimen(path):
    """ From filepath, get specimen (assuming format is like CM_M_01_Back.bil: species_sex_specimen_facing) """
    na = GetFilename(path)
    pattern = re.compile(r"(?<=^).*(?=_[A-z]*$)") # this gets the specimen
    res = re.findall(pattern, na)
    return res

def GetSpecies(path):
    """ From filepath, get species (assuming format is like CM_M_01_Back.bil: species_sex_specimen_facing) """
    na = GetFilename(path)
    pattern = re.compile(r"^[^0-9]*(?=_)") # this gets the species
    res = re.findall(pattern, na)
    return res

def CheckFileMatch(matfiles, hdrfiles, bilfiles):
    """ Check matching order between the items in 3 lists: used to check that metadata, hdr files, and hsi all refer to the same base image """

    for i, k in enumerate(bilfiles):
        for j in range(i+1, len(bilfiles)):
            if GetFilename(k) == GetFilename(bilfiles[j]):
                print( k )
                print( bilfiles[j] )
                raise Exception('Duplicates filenames exist in the datafiles, these must be fixed, examples printed')
        
    for i, j, k in zip(matfiles, hdrfiles, bilfiles):
        if not GetFilename(i) == GetFilename(j) == GetFilename(k): # if files don't match order error
            print( GetFilename(i) )
            print( GetFilename(j) )
            print( GetFilename(k) )
            raise Exception('Order of mat, hdr, bil files dont match - this must be fixed')



    #if len(matfiles) == len(hdrfiles) == len(bilfiles):
    #    raise Exception('Length of mat, hdr, bil files dont match - this must be fixed')
            
    print('Filenames match, okay to continue')

################ Functions to generate human and avian color values from spectra #############

def PlotFalseRGB(img, r = 600., g = 550., b = 425., standard_100 = np.nan, plotme = False, normalize_max = .2):
    """ Plot a rgb version of a hyperspectral image by selection of only 3 bands (note: not using CIE or other color space convserions)

    Arguments:
    img - hyperspectral image img (generated with spectral.io.envi.open), 
    r, g, b - float wavelength bands for r, g, and b color channels respectively, function takes the closest bands to these values in img
    standard_100 - optional np.array of measurement of 99% reflectance standard with which to convert the hsi to reflectance(with items equal to the number of bands in the hyperspectral image),
    plotme - boolean if True returns a figure and axis as well as the rgb image, otherwise return just zeros and the rgb image. 
    normalize_max - (float 0-1) determines the 'white point' of the resulting image, i.e. a reflectance that produces a saturated color for that channel. Lower 
    values to deal with images that look oversaturated (because few spectra approach 100% reflectance) 
    
    """

    # get image wavelength centers
    owl = img.bands.centers

    # find the idx of the closest bands in image
    r = np.argmin(np.abs(np.array(owl)-r)).astype(int)
    g = np.argmin(np.abs(np.array(owl)-g)).astype(int)
    b = np.argmin(np.abs(np.array(owl)-b)).astype(int)
    
    # find conversion factor, make into reflectance
    if np.isnan(standard_100).all():
        refs = np.array((1, 1, 1)) # if np.na passed here, assume image is already reflectance 
    else:
        refs = np.array((standard_100[r], standard_100[g], standard_100[b]))
    r = img[:,:,int(r)] / refs[0]
    g = img[:,:,int(g)] / refs[1]
    b = img[:,:,int(b)] / refs[2]

    # set normalization for contrast
    norm = Normalize(vmin=0., vmax=normalize_max)
    im = norm(np.concatenate((r,g,b), axis = 2))
    # im = np.concatenate((r,g,b), axis = 2) # use this instead to omit normalization, (same as normalize_max = 1) but since this 
    # makes 100% reflectance 255,255,255 it tends to underexpose resulting images

    if plotme == True:
        fig, ax = plt.subplots(1, 
                              figsize=(12,8), dpi=100, facecolor='w', edgecolor='k')
        ax.imshow(im)
        return fig, ax, im    
    else:
        return 0, 0, im

def GetColors(sel, wl, color_scale = 2):
    """ Uses the spectra to generate XYZ trisimulus and sRGB (brightened) colors- using the colour package.
    
    Arguments:
    sel - sample x wavelength np.array, 
    wl - np.array of the band centers of sel
    color_scale - float multiplier for the sRGB color, to help make bright visible figure colors
    
    """

    ## Different versions of color package seem to have widely differing syntax, and efficiency. In v0.4.1, the following code
    ## runs very slowly.
    print('Generating colors')
     
    if np.ndim(sel)==1: # if just one spectrum (colour package awkward in this respect)
        sample = colour.SpectralDistribution(data = sel.T, domain = wl)
    else:
        sample = colour.MultiSpectralDistributions(data = sel.T, domain = wl)

    print('Sample in colour-science format')
    
    # use 1931 CIE standard, and D65 light, see colour package for other options
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    illuminant = colour.SDS_ILLUMINANTS["E"] # ["D65"] # use ideal illuminants
    illuminant_cc = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['E']
    
    if np.ndim(sel)==1: # if just one spectrum (colour package awkward in this respect)
        # Since we're interested in getting XYZ, Lab, -and- sRGB, we don't just use
        # automatic conversion: (sd, "Spectral Distribution", "sRGB", sd_to_XYZ={"illuminant": SDS_ILLUMINANTS["FL2"]})
        XYZ = colour.sd_to_XYZ(sample, cmfs, illuminant, method="Integration")
        # print('Got XYZ')
    else: 
        XYZ = colour.msds_to_XYZ(sample, cmfs, illuminant, method="Integration")
        # print('Got XYZ')

    Lab = colour.XYZ_to_Lab(XYZ, illuminant_cc)
    # print('Got Lab')
    sRGB = colour.XYZ_to_sRGB((XYZ / 100) * color_scale, illuminant_cc) 
    # print('Got sRGB') # note that XYZ_to_sRGB expects XYZ in scale [0-1], not [0-100]. If we supply k to sd_to_XYZ, we'd get XYZ in that scale. "When 
    # is set to a value other than None, the computed CIE XYZ tristimulus values are assumed to be absolute and are thus converted from percentages by a final 
    # division by 100." Rather than do the division here, we could find k that results in conversion 1 * E -> XYZ = 1 rather than conventional 100. 
    # color_scale is a multiplier to the rgb values to get colors to show up nicely in plublication.
        
    # clamp rgb colors
    sRGB[sRGB>1]=1
    sRGB[sRGB<0]=0

    print('Colors generated')

    return XYZ, sRGB, Lab

def Spec2usml(sel, wl):
    """ 
    Converts spectra to u, s, m, l stimulus values using multiplication and integration by summation, currently hard coded to an average VS avian 
    model (extracted from pavo, https://cran.r-project.org/web/packages/pavo/index.html), but easily extended to other tetrahedral models. Currently ignores 
    illumination and background - mathematically equivalent to assuming ideal for both. 
    
    Arguments:
    sel - sample x wavelength np.array of reflectances
    wl - (np.array) of the band centers for sel

    Returns stimulation of each of the cone types for each spectrum, as well as relative u, s, m, l stimulus values
    and the corresponding x, y, z values in a tetrahedral colorspace

    """
    
    if len(sel.shape) < 2: # if one dimensional, fix to sample (1) x wl
        sel = sel[None,:]
    
    # load sensitivity
    df = pandas.read_csv( '../dat/Sensi/Avg_v.csv', index_col=None).to_numpy()
    s_wl = df[:,0]
    s_df = df[:,1:]
    
    # interpolate sensitivity to wavelengths of the sample, be sure that wavelengths of sample fall within bounds of sensitivity
    # for simplicity here, we don't
    s_wl, s_df = InterpSample(s_wl, s_df.T, wlnew = wl)
    s_df = np.divide(s_df, np.sum(s_df, axis = 1)[:,None]) # normalize to area = 1
    
    if not np.all(wl == s_wl):
        raise Exception('Visual sensitivity and spectra wavelegths do not match')
    
    # get usml out
    usml = np.zeros((sel.shape[0], s_df.shape[0]))
    for i in range(s_df.shape[0]):
        usml[:,i] = np.sum(np.multiply(s_df[i,:], sel), axis = 1)
    
    if len(usml.shape) < 2: # if only 1 sample, maintain shape
        usml = usml[None,:]

    # get relative usml out
    relusml = np.divide(usml, np.sum(usml,1)[:,None]) # should check this
    print('Got USML + relative USML')
    
    txyz = GetTxyz(relusml).T
    
    return usml, relusml, txyz


def GetTxyz(usmlin):
    """
    Get tetracolorspace XYZ from relative usml values. This code inspired by TetraColorSpace implementation (https://www.marycstoddard.com/software)
    
    Arguments:
    usml - sample x 4 np.array of relative usml stimulation values
    
    """
    

    X = (1-2*usmlin[:,1] - usmlin[:,2] - usmlin[:,0])/2 * math.sqrt(3/2);
    Y = (-1 + 3*usmlin[:,2]+usmlin[:,0]) / (2*math.sqrt(2));
    Z = (usmlin[:,0]) - 1/4;

    txyz = np.array([X, Y, Z])
    return txyz

def GetThetaPhi(x,y,z):
    """ This code is inspired by Pavo implementation (https://cran.r-project.org/web/packages/pavo/index.html)
    generates theta and phi for given tetrahedral xyz coordinates 
    
    Arguments:
    x, y, z np.arrays of the coordinates of points in tetrahedral color space """
    
    r_vec = np.sqrt(x * x + y * y + z * z)
    theta = np.arctan2(y, x)
    phi = np.arcsin(z / r_vec)
    return theta, phi

def GetRNL(array, ratios = np.array((1,1,1,2)), weber_ratio = 0.1, logarithm = False):
    """ 
    Generate RNL coordinates, as well as saturation (following following van den Berg et al, 2019)
    Note that we follow formulas in Renoult 2017 for tetrachromats (note 0 index in python lowers formula indices by 1)
    arguments: 
    array - sample x 4 relative usml np.array, 
    ratios - np.array ratios of photoreceptor density for each channel. Default cone ratios (np.array((1,1,1,2))) from Columbia livia (Vorobyev & Osorio 1998)
    weber ratio - chromatic weber ratio to use, default 0.1
    logarithm - boolean whether to log the input (Weber–Fechner law)

    Can easily get pairwise distance from this, similar to dS with:
    from scipy.spatial.distance import pdist
    pdist(np.vstack((x,y,z)).T) 
    
    """

    if np.logical_or(array.shape[1] != 4, len(ratios) != 4):
        raise Exception("Both usml array and list of cone ratios must have 4 channels")

    if not np.all(np.sum(array, axis = 1) - 1 < .001): # if not close to unit
        raise Exception("GetRNL expects relative cone catch values, for chromatic channels only (I.e. all sum to 1)")

    weber = weber_ratio / np.sqrt(ratios / ratios.max()) # weber = this is e_i in Renoult
    
    # do we use log or linear logarithmic (Weber–Fechner law) version of the model, see Renoult (A1.5) and (A1.6)
    if logarithm == False:
        sn = array
    elif logarithm == True:
        sn = np.log(array) 
    
    # since squared versions are used a few times, save them to simplify formulas
    weber_sqr = weber**2
    weber_12_sqr = (weber[1] * weber[2])**2
    weber_23_sqr = (weber[2] * weber[3])**2
    weber_13_sqr = (weber[1] * weber[3])**2
    weber_123_sqr = (weber[1] * weber[2] * weber[3])**2
    weber_023_sqr = (weber[0] * weber[2] * weber[3])**2
    weber_013_sqr = (weber[0] * weber[1] * weber[3])**2
    weber_012_sqr = (weber[0] * weber[1] * weber[2])**2
    
    # make x
    x_web = np.sqrt(1 / weber_sqr[2] + weber_sqr[3])
    x = x_web * (sn[:,3] - sn[:,2])
    
    # make y
    y_web =  np.sqrt(weber_sqr[2] + weber_sqr[3] / 
                weber_12_sqr + weber_13_sqr + weber_23_sqr) 
    y = y_web * (sn[:,1] - (sn[:,3] * (weber_sqr[2] / weber_sqr[2] + weber_sqr[3])  + sn[:,2] * (weber_sqr[3] / weber_sqr[2] + weber_sqr[3])))
    
    # make z
    z_A = np.sqrt( weber_23_sqr + weber_13_sqr + weber_12_sqr /
                  weber_123_sqr + weber_023_sqr + weber_013_sqr + weber_012_sqr )
    z_a = weber_12_sqr / weber_23_sqr + weber_12_sqr + weber_13_sqr        
    z_b = weber_13_sqr / weber_23_sqr + weber_12_sqr + weber_13_sqr
    z_c = weber_23_sqr / weber_23_sqr + weber_12_sqr + weber_13_sqr
    z = z_A * (sn[:,0] - (z_a * sn[:,3] + z_b * sn[:,2] + z_c * sn[:,1]))

    # following van den Berg et al, 2019 - euclidean distance to origin = saturation
    sat = x**2 + y**2 + z**2
    
    return x, y, z, sat

################ Helper to make a purple-blue-green-red colormap ############

def GetCustomColormap(plotme = False):
    """ Convenience function to generate a custom heatmap that approximately reflects human visible spectrum purple-blue-green-red returns a colormap 
    for use with matplotlib plotting functions. Some handy functions directly from https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72 """
    
    def hex_to_rgb(value):
        '''
        Converts hex to rgb colours
        value: string of 6 characters representing a hex colour.
        Returns: list length 3 of RGB values'''
        value = value.strip("#") # removes hash symbol if present
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


    def rgb_to_dec(value):
        '''
        Converts rgb to decimal colours (i.e. divides each value by 256)
        value: list (length 3) of RGB values
        Returns: list (length 3) of decimal values'''
        return [v/256 for v in value]

    def get_continuous_cmap(hex_list, float_list=None):
        ''' creates and returns a color map that can be used in heat map figures.
            If float_list is not provided, colour map graduates linearly between each color in hex_list.
            If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 

            Parameters
            ----------
            hex_list: list of hex code strings
            float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

            Returns
            ----------
            colour map'''
        rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
        if float_list:
            pass
        else:
            float_list = list(np.linspace(0,1,len(rgb_list)))

        cdict = dict()
        for num, col in enumerate(['red', 'green', 'blue']):
            col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
            cdict[col] = col_list
        cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
        return cmp

    import matplotlib.colors as mcolors

    hex_list = ['#FF00FF', '#0000FF', '#40E0D0', '#008080', '#008000', '#FFFF00', '#FFA500', '#FF0000']
    custom_map = get_continuous_cmap(hex_list)

    if plotme == True:
        xx, yy = np.mgrid[-5:5:0.05, -5:5:0.05]
        zz = (np.sqrt(xx**2 + yy**2) + np.sin(xx**2 + yy**2))
        fig, ax = plt.subplots(1,1)
        im = ax.imshow(zz, cmap=custom_map)
        fig.colorbar(im)
        ax.yaxis.set_major_locator(plt.NullLocator()) # remove y axis ticks
        ax.xaxis.set_major_locator(plt.NullLocator()) # remove x axis ticks
        plt.show()
    
    return custom_map