################ Functions for tetrahedral space plots ################
# note points3dColor expects integer rgb colors
# example:
# from plotly.subplots import make_subplots
# fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
# tetra.tetra3d(fig)
# tetra.points3dColor(fig, xyz, sRGB = cols, grouping = None)
# fig.show()

# or:
# fig = make_subplots(rows=1, cols=2, specs = [[{"type": "scene"}, {"type": "scene"}]])
# tetra.tetra3d(fig, row = 1, col = 1, label_size = 20)
# tetra.tetra3d(fig, row = 1, col = 2, label_size = 10)
# fig.show()

import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde # used in Get3dKDE for kernel density estimation

def tetra3d(fig, opacity = 0.25, row = 1, col = 1, add_labels = True, labels = ['U', 'S', 'M', 'L'], label_size = 10):
    
    """  Add a 3d tetrahedron to a plotly figure, note that this is set up to allow subplot use with row and columns defined.
    Arguments:
        fig - plotly figure handle
        opacity - transparency of the tetrahedral faces
        row, col - coordinates of the current subplot of the figure, default 1, 1 works for non-subplotted figure
        add_labels - boolean whether to add labels listed in 'labels' to the vertexes of the tetrahedron
        labels - list of str labels for vertexes, in order of increasing wavelength of peak sensitivity
        label_size - size of the text labels
    """
    
    # define tetra vertice coordinates
    XYZ = np.array([[0, 0, .75],  
      [-.61237, -.35355, -.25], 
      [0, .70711, -.25], 
      [.61237, -.35355, -.25]])

    # plot tetrahedron
    fig.add_trace(go.Mesh3d(
            x=XYZ[:,0],
            y=XYZ[:,1],
            z=XYZ[:,2],
            colorbar_title='z',
            # Intensity of each vertex, which will be interpolated and color-coded
            vertexcolor =  [ 'rgb(255, 0, 255)', 'rgb(0, 0, 255)', 'rgb(0, 255, 0)', 'rgb(255, 0, 0)' ],
            # intensitymode='cell',
            # i, j and k give the vertices of triangles
            # here we represent the 4 triangles of the tetrahedron surface
            i=[0, 0, 0, 1],
            j=[1, 2, 3, 2],
            k=[2, 3, 1, 3],
            name='y',
            showscale=True,
            opacity = opacity,
        ), row=row, col=col)

    # add edges for the vertices
    AddTriEdges(fig, XYZ, row, col)
    
    # add labels to vertices
    if add_labels == True:
        AddVertexLabels(fig, row, col, XYZ, labels = labels, label_size = label_size)
    
    fig.layout.scene.camera.projection.type = "orthographic"
    

def AddVertexLabels(fig, row, col, XYZ, labels = ['U', 'S', 'M', 'L'], label_size = 10, label_color = "black"):
    """ Function to label the vertices of tetrahedral plot in plotly figure.
    Arguments:
    fig, row, col - Plotly figure handle, and row and column coordinates (1,1 works for un-subplotted figure)
    XYZ - 3D coordinates for the vertex label positions
    labels - list of str labels for the vertexes
    label_size, label_color - self explanatory
    """

    fig.update_scenes(row = row,
        col = col,
        annotations=[
        dict(x=XYZ[0,0],
            y=XYZ[0,1],
            z=XYZ[0,2],
            text=labels[0],
            xanchor="auto",
            xshift=0,
            yshift=0,
            opacity=1,
            font=dict(
                family="Times New Roman",
                size=label_size,
                color=label_color)),
        dict(x=XYZ[1,0],
            y=XYZ[1,1],
            z=XYZ[1,2],
            text=labels[1],
            xanchor="auto",
            xshift=0,
            yshift=0,
            opacity=1,
            font=dict(
                family="Times New Roman",
                size=label_size,
                color=label_color)),
        dict(x=XYZ[2,0],
            y=XYZ[2,1],
            z=XYZ[2,2],
            text=labels[2],
            xanchor="auto",
            xshift=0,
            yshift=0,
            opacity=1, 
            font=dict(
                family="Times New Roman",
                size=label_size,
                color=label_color)),
        dict(x=XYZ[3,0],
            y=XYZ[3,1],
            z=XYZ[3,2],
            text=labels[3],
            xanchor="auto",
            xshift=0,
            yshift=0,
            opacity=1, 
            font=dict(
                family="Times New Roman",
                size=label_size,
                color=label_color))])


def AddTriEdges(fig, XYZ, row, col):
    """ Utility function to add lines for the vertex edges for a tetrahedral plot in plotly 
    Arguments:
    fig, row, col - Plotly figure handle, and row and column coordinates (default 1,1 works for un-subplotted figure) 
    XYZ - 3D coordinates for the vertex positions
    """
    
    lwidth = 1 # linewidth
    faces = np.array([[0, 1, 2], # define planes
               [0,2,3],
               [0,1,3],
               [1,2,3]])
    x, y, z  = XYZ.T
    i, j, k = faces.T
    #plot surface triangulation
    tri_vertices = XYZ[faces]
    Xe = []
    Ye = []
    Ze = []
    for T in tri_vertices:
        Xe += [T[k%3][0] for k in range(4)] + [ None]
        Ye += [T[k%3][1] for k in range(4)] + [ None]
        Ze += [T[k%3][2] for k in range(4)] + [ None]
    fig.add_trace(go.Scatter3d(x=Xe,
                         y=Ye,
                         z=Ze,
                         mode='lines',
                         name='',
                         showlegend=False,
                         line=dict(color= 'rgb(40,40,40)', width=lwidth)), row = row, col = col);



def points3dColor(fig, xyz, sRGB = None, grouping = None, row = 1, col = 1, msize = 2):
    """ Scatter points in 3d plotly figure, colored by a vector of sRGB values, or by factor grouping
    uses xyz dataset (an n by 3 numpy array), along with optional rgb for every point and optional grouping vector
    set up for subplotting in plotly with row and column.
    
    Arguments:
    fig, row, col - Plotly figure handle, and row and column coordinates (default 1,1 works for un-subplotted figure) 
    xyz - sample x 3 float coordinates for points in tetrahedral colorspace
    sRGB - optional sample x 3 float RGB colors (0-1) for points
    grouping - optional sample x 1 numpy array of string or integer groupings
    msize - marker size
    
    for example:
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
    tetra.tetra3d(fig)
    xyz = np.array([[0, 0, 0], [0, 0, .1], [0, 0, .2], [0, 0, .3], [0, 0, .4], [0, 0, .5]])
    sRGB = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1]], dtype = 'float')
    grouping = np.array(['blue', 'teal', 'other', 'other', 'other', 'other'])
    tetra.points3dColor(fig, xyz, sRGB = sRGB, grouping = grouping, row = 1, col = 1, msize = 10)
    fig.show()
    """
    
    assert xyz.shape[1]==3, "xyz should be a sample x 3 numpy array of tetrahedral coordinates"
    assert xyz.shape == sRGB.shape, "sRGB should be a sample x 3 (R, G, B) numpy array of colors (0-1)"
    sRGB[sRGB >= 1] = .9999 # plotly does not colors like [0, 0, 1] which plots as black
        
    if grouping is not None:
        if sRGB is not None: # plot rgb, but also have grouping - we plot through each group, so that we still have access to grouping for any legends
            markcol=[f'rgb({sRGB[k,0]}, {sRGB[k,1]}, {sRGB[k,2]})' for k in range(sRGB.shape[0])]
            for i in np.unique(grouping):
                ssRGB = sRGB[grouping==i,:]
                markcol=[f'rgb({ssRGB[k,0]}, {ssRGB[k,1]}, {ssRGB[k,2]})' for k in range(ssRGB.shape[0])]
                fig.add_trace(go.Scatter3d(x=xyz[grouping==i,0], y=xyz[grouping==i,1], z=xyz[grouping==i,2], 
                  mode='markers', 
                  name = i,
                  marker=dict(color = markcol, size=msize)
                 ), row = row, col = col)
            fig.update_layout(legend= {'itemsizing': 'constant'})
            
    if grouping is not None:
        if sRGB is None: # we have only grouping, plot by that
            for i, j in zip(np.unique(grouping), range(len(np.unique(grouping)))):
                sRGB = np.array(color_dict.get(i)) * 255
                sRGB = sRGB.astype('int')
                markcol=[f'rgb({sRGB[0]}, {sRGB[1]}, {sRGB[2]})']
                fig.add_trace(go.Scatter3d(x=xyz[grouping==i,0], y=xyz[grouping==i,1], z=xyz[grouping==i,2], 
                  mode='markers', 
                  name = i,
                  marker=dict(size=msize, 
                              color = np.repeat(markcol, np.sum(grouping==i))) # cs[j]
                 ), row = row, col = col)
            fig.update_layout(legend= {'itemsizing': 'constant'})
            
    if grouping is None:
        if sRGB is not None: # if we have srgb but don't have grouping, just add them all in one go
            markcol=[f'rgb({sRGB[k,0]}, {sRGB[k,1]}, {sRGB[k,2]})' for k in range(sRGB.shape[0])]
            fig.add_trace(go.Scatter3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], # DONT KNOW IF WORKING
                          mode='markers', 
                          marker=dict(size=msize, color = markcol)
                         ), row = row, col = col)     

    if grouping is None:
        if sRGB is None: # if we have neither rgb or grouping, just plot black points
            fig.add_trace(go.Scatter3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], 
                          mode='markers', marker=dict(size=msize, color = 'black')), 
                          row = row, col = col)

def Get3dKDE(pltd, res = 65, bw_method = 'scott'):
    """ Generate 3D kernel density from x, y, and z positions (an sample by 3 numpy array of coordinates in tetrahedral color space), 
    returns the sample grid, and densities for each point
    
    Arguments:
    pltd - sample x 3 numpy float array of x, y, and z coordinates in tetrahedral color space
    res - resolution of the grid of 3d density samples eventually created from just under to just over the max values of x, y, and z
    bw_method see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

    # for example: xi, yi, zi, density = Get3dKDE(pltd, res = 40j, bw_method = 'scott') 
    """
    
    # get position
    x = pltd[:,0] 
    y = pltd[:,1] 
    z = pltd[:,2] 
        
    # Old version, uses bandwidth
    print('Doing kde')
        
    # define resolution of grid sampling, this this takes complex number - that makes it the number of steps, rather than the interval
    densitygridS = complex(res)

    # transpose for kde
    xyz = np.vstack([x,y,z])

    # apply kde
    old = True
    if old == True:
        # https://stackoverflow.com/questions/25286811/how-to-plot-a-3d-density-map-in-python-with-matplotlib
        print('Using stats.guassian_kde')
        kde = gaussian_kde(xyz, bw_method)
    else:
        # do with scikitlearn
        print('Using sklearn.KernelDensity')
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)

    print('Calculating KDE on points')
    # Evaluate kde on a grid
    xmin, ymin, zmin = x.min()-0.1, y.min()-0.1, z.min()-0.1
    xmax, ymax, zmax = x.max()+0.1, y.max()+0.1, z.max()+0.1
    xi, yi, zi = np.mgrid[xmin:xmax:densitygridS, ymin:ymax:densitygridS, zmin:zmax:densitygridS]
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]]) 
    density = kde(coords).reshape(xi.shape)

    # density = density/density.max() # possible to normalize the kde
    return xi, yi, zi, density

    

def plot3dKDE(fig, xi, yi, zi, density, wcolor = 'green', na = None, row = 1, col = 1):
    """ Adds a trace of a 3d kernel density calculation to a plotly 3d plot as a volume
    
    Arguments:
    fig, row, col - Plotly figure handle, and row and column coordinates (default 1,1 works for un-subplotted figure) 
    xi, yi, zi, density - kernel density grid coordinates and density estimates (see Get3dKDE())
    wcolor - the color of the volume
    na - optional label for the volume for legends

    # for example:
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
    tetra.tetra3d(fig)
    xi, yi, zi, density = tetra.Get3dKDE(txyz, res = 40j, bw_method = 'scott')
    density = density/density.max()
    tetra.plot3dKDE(fig, xi, yi, zi, density, wcolor = 'blue', na = 'this_kde', row = 1, col = 1)        
    fig.show() """
    
    fig.add_trace(go.Volume(
    x=xi.flatten(),
    y=yi.flatten(),
    z=zi.flatten(),
    value=density.flatten(),
    colorscale=[[0, wcolor], [0.5, wcolor], [1.0, wcolor]],
    isomin=0.1,
    isomax=1,
    opacity=1, # max opacity # needs to be small to see through all surfaces
    opacityscale=[[0, 0.2], [0.5, 0.5], [1, 1]],
    surface_count=5, # needs to be a large number for good volume rendering
    name = na,
    showlegend=True,
    showscale=False,
    ), row = row, col = col)

def plot3dKDE_iso(fig, xi, yi, zi, density, wcolor = 'green', na = None, row = 1, col = 1, level = 0.05):
    """ Adds a trace of a 3d kernel density calculation to a plotly 3d plot as a isosurface

    Arguments:
    fig, row, col - Plotly figure handle, and row and column coordinates (default 1,1 works for un-subplotted figure) 
    xi, yi, zi, density - kernel density grid coordinates and density estimates (see Get3dKDE())
    wcolor - the color of the volume
    na - optional label for the volume for legends
    level - the density level to enclose in the surface. By default (0.05) , this draws one isosurface at 5% density cut-off, enclosing 95% 
    of the density. If you want more than one level directly use plotly, or modify function.

    # for example:
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
    tetra.tetra3d(fig)
    xi, yi, zi, density = tetra.Get3dKDE(txyz, res = 40j, bw_method = 'scott') # cs[j]
    density = density/density.max()
    tetra.plot3dKDE_iso(fig, xi, yi, zi, density, wcolor = 'blue', na = 'this_kde', row = 1, col = 1)        
    fig.show()
    """
    
    fig.add_trace(go.Isosurface(
    x=xi.flatten(),
    y=yi.flatten(),
    z=zi.flatten(),
    value=density.flatten(),
    colorscale = [[0, wcolor], [1, wcolor]],
    isomin=level,
    isomax=level,
    surface_count=1,
    name = na,
    opacity = 0.5,
    showscale = False,
    caps=dict(x_show=False, y_show=False)
    ), row = row, col = col)

