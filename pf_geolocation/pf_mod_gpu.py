# Test code from https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb

from numpy.random import uniform
from numpy.random import randn
from scipy.spatial import cKDTree
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule
import pycuda.driver as driver
import numpy as np



def nearest(X, y):
    dist = np.sum((X-y)**2,axis=1)
    #idx = np.where(dist==min(dist))
    #return (X[idx][0], idx[0][0])
    idx = np.argmin(dist)
    return (X[idx], idx)

def isintriangle(xt,yt,x0,y0):
    res = False
    f1 = (y0-yt[0])*(xt[1]-xt[0]) - (x0-xt[0])*(yt[1]-yt[0])
    f2 = (y0-yt[2])*(xt[0]-xt[2]) - (x0-xt[2])*(yt[0]-yt[2])
    f3 = (y0-yt[1])*(xt[2]-xt[1]) - (x0-xt[1])*(yt[2]-yt[1])
    if(f1*f3 >= 0.0 and f3*f2 >= 0.0):
            res = True
    return res

def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles







def update(particles, weights, iterr, iObsLh, fvcom):
    import matplotlib.tri as mtri
    import scipy.io


    weights = np.zeros_like(weights)

    triang = mtri.Triangulation(fvcom.x, fvcom.y, (fvcom.tri-1))
    interp = mtri.LinearTriInterpolator(triang, iObsLh)
    
    weights = interp(particles[iterr, :, 0], particles[iterr, :, 1]).data
    weights[np.isnan(weights)] = 0

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize
    
    
    
    return weights

def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var


# def simple_resample(particles, weights):
#     N = len(particles)
#     cumulative_sum = np.cumsum(weights)
#     cumulative_sum[-1] = 1. # avoid round-off error
#     indexes = np.searchsorted(cumulative_sum, random(N))

#     # resample according to indexes
#     particles[:] = particles[indexes]
#     weights[:] = weights[indexes]
#     weights /= np.sum(weights) # normalize


# def neff(weights):
#     return 1. / np.sum(np.square(weights))

def resample_from_index(particles, total_weights, weights, indexes):
    particles[:,:,:] = particles[:,indexes,:]
    total_weights[:, :] = total_weights[:, indexes]
    weights = weights[indexes]
    weights /= np.sum(weights)

    return weights













