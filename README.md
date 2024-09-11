# Hyperspectral Birds-Of-Paradise
Code and examples for "Hyperspectral imaging in animal coloration research: a user-friendly pipeline for image generation, analysis and integration with 3D modeling."
 
 Benedict G. Hogan, Mary Caswell Stoddard. 2024.
 
Installation requirements:
-----------------------

Assumes working installation of anaconda (but use whatever environment manager you prefer, and modify the below to suit). <br />

Open anaconda prompt (on windows, want anaconda powershell prompt) and navigate to "../HyperSpectralCluster/", using: <br />
**cd C:/PathYouClonedGitHubTo/HyperBirdOfParadise/** <br />

Run: <br />
**conda env create --file=hsi_environment.yml**  <br />
Or if you want to name the environment something other than hsi, use: <br /> **conda env create --name --file=hsi_environment.yml** <br />
Note that the packages installed as requirements in hsi_environment.yml includes packages that are required to replicate the paper, but are not strictly required to use the hsi package code found in /hsi/. That said, you may find many of the packages useful if you use the code from the paper as inspiration for your own analyses and plots.<br />

Then activate the environment containing the required packages by running: <br />
**conda activate hsi** <br />

Then open jupyter notebook by running: <br />
**jupyter notebook**

You can then navigate the scripts and run jupyter notebook files etc.

Useful functions, hyperspectral exploration:
-----------------------

We use jupyter notebooks to give examples of loading and manipulating hyperspectral images in the .ipynb notebook sample_examples.ipynb. This would be the best place to start. Some directories will need updating to reflect the location to which you download the dataset/s. For simple_examples, simply update "datadir" to reflect the location of the (un-zipped) gouldian finch hyperspectral image and header file. 

Generic functions written to help deal with and plot hyperspectral data are found in the python package ./hsi. This can be imported into python similarly to other packages, for instance "import hsi.utils as utils", if your python current directory contains the /hsi/ folder. Most are not terribly complex so we encourage you to explore the .py files if you have questions.

These functions make use of the very useful libraries colour (https://www.colour-science.org/) and Spectral python (https://www.spectralpython.net/) among many others.

Replicating paper:
-----------------------

Jupyter notebook bop_notebook.ipynb runs the analysis found in the main paper, generating first a set of pickled files (one for each plumage patch across specimens and species), which contain all the samples and various embeddings, distances, and color representations of the samples. The code then generates summary plots of various kinds. Some directories will need updating to reflect the location to which you download the dataset/s. For bop_notebook, update "lookin" to list the folder/s in which all of the bird-of-paradise hyperspectral images and header files are, as well as the (un-zipped) folder containing MATLAB metadata files (/matmask/)/.

This analysis uses functions that are packaged into bop_sampling, bop_plotting and bop_analysis. See comments in code there for explanations of those functions. However, all code in those packages are written specifically for the analysis used in the paper. So to replicate functionality for your own work, please use that code as inspiration.

Data locations:
-----------------------

Please find the data used in the paper (hyperspectral images, hyperspectral .hdr files, and .mat files containing metadata including binary masks for each patch, standard location and size, etc) on Dryad (10.5061/dryad.j0zpc86nf), and data used in the simple_examples.ipynb (hyperspectral image of gouldian finch, and associated .hdr files) at the same location.
