# Test code from https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb

from numpy.random import uniform
from numpy.random import randn
from scipy.spatial import cKDTree
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

def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 2))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    return particles

def predict(particles, hdiff, iterr, nsub, fvcom, dt=1.):

    xv = fvcom.x
    yv = fvcom.y
    nv = (fvcom.tri-1).T
    xc = fvcom.xc
    yc = fvcom.yc
    ntve = fvcom.ntve
    nbve = fvcom.nbve
    nodes = np.vstack([xv,yv]).T

    N = len(particles[0])
    stat = np.ones(N,dtype=np.int32)

    x = particles[iterr-1, :, 0]
    y = particles[iterr-1, :, 1]

    import matplotlib.pyplot as plt
    # plt.triplot(xv, yv, nv.T)

    for i in range(nsub):

        print("    Moving particles in substep "+str(i+1)+"/"+str(nsub)+"...")

        deltat = 86400.0 / nsub
        tscale = (2*deltat*hdiff)**0.5;

        # horizontal random walk
        x2 = x
        y2 = y

        x = x + (randn(N) * tscale)
        y = y + (randn(N) * tscale)

        # find nearest node
        t_nodes = cKDTree(nodes)
        pt = np.vstack([x,y]).T
        _,minloc = t_nodes.query(pt, k=2)

        

        for i in range(N):
            # find if particle is in any of the cells surrounding the two nearest nodes
            nbve_unique = np.unique(nbve[:,minloc[i]])[1:]-1
            for cell_idx in nbve_unique:
                # cell_idx = nbve_unique[cell]-1
                xtri = xv[nv[:,cell_idx]]
                ytri = yv[nv[:,cell_idx]]
                if isintriangle(xtri,ytri,pt[i,0],pt[i,1]):
                    stat[i] = 1
                    break
                else:
                    stat[i] = 0
            if stat[i] == 0: # for out-of-domain particles redo random walk within the same substep until the particle is inside domain
                while True:
                    x[i] = x2[i] + (randn() * tscale)
                    y[i] = y2[i] + (randn() * tscale)
                    _,minloc_r = t_nodes.query([x[i], y[i]], k=2)
                    nbve_unique = np.unique(nbve[:,minloc_r])[1:]-1
                    # print(minloc_r)
                    for cell_idx in nbve_unique:
                        # cell_idx = nbve[cell,minloc_r]-1
                        xtri = xv[nv[:,cell_idx]]
                        ytri = yv[nv[:,cell_idx]]
                        if isintriangle(xtri,ytri, x[i], y[i]):
                            stat[i] = 1
                            break
                        else:
                            stat[i] = 0
                    if stat[i] == 1:
                        # print("[DEBUG] An out-of-domain particle was made sure to stay within domain during random walk.")
                        # import ipdb; ipdb.set_trace()
                        break
                    # else:
                        # print("[DEBUG] Iteration failed")
                        # plt.plot(x[i],y[i],'rx')
                        # import ipdb; ipdb.set_trace()

        # update particle state and reset particles on land 
        x[stat==0] = x2[stat==0]
        y[stat==0] = y2[stat==0]

    # dump end-of-day locations
    particles[iterr, :, 0] = x
    particles[iterr, :, 1] = y



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













