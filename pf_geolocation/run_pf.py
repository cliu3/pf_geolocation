from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io
import netCDF4
from pf_mod import *
from my_project import *
from filterpy.monte_carlo import systematic_resample
import os.path
import os
import sys
sys.path.append(os.getcwd())
from likelihood import *
from config import *

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def main():
    """
    This is the main function of the particle filter geolocation package.
    """
    hdiff_coef_in_km2_per_day = np.array([hdiff_high, hdiff_moderate, hdiff_low])  # array of 3 [HIGH MODERATE LOW]

    # load FVCOM GOM mesh
    grid_nc = netCDF4.Dataset(fvcom_grid)
    fvcom = {
        "x": grid_nc.variables['x'][:],
        "y": grid_nc.variables['y'][:],
        "tri": grid_nc.variables['nv'][:].T,
        "xc": grid_nc.variables['xc'][:],
        "yc": grid_nc.variables['yc'][:],
        "ntve": grid_nc.variables['ntve'][:], # number of elems surrounding each node
        "nbve": grid_nc.variables['nbve'][:], # elems surrounding each node
    }
    fvcom=objectview(fvcom)
    # mat=scipy.io.loadmat(fvcom_tidaldb,squeeze_me=True, struct_as_record=False)
    # fvcom=mat['fvcom']

    for tagid in tagid_list:
        # load tag
        tag=scipy.io.loadmat(path_to_tags+'/'+str(tagid)+'_raw.mat',squeeze_me =False,struct_as_record=True)
        tag=tag['tag'][0,0]
        dnum=tag['dnum'][:,0]
        temp=tag['temp'][:,0]
        depth=tag['depth'][:,0]
        dnum=dnum-678942
        release_lon = tag['release_lon'][0,0]
        release_lat = tag['release_lat'][0,0]
        [release_x, release_y] = my_project(release_lon, release_lat, 'forward')
        recapture_lon = tag['recapture_lon'][0,0]
        recapture_lat = tag['recapture_lat'][0,0]
        [recapture_x, recapture_y] = my_project(recapture_lon, recapture_lat, 'forward')

        tagname = str(tagid)+'_'+tag['tag_id'][0]

        print('Processing tag: '+tagname+'...')

        #####################################
        # load observation likelihood data
        #####################################
        obslh_fname = lhpath+'/ObsLh'+tagname+'.mat'
        if (not os.path.isfile(obslh_fname) ) or ( not use_existing_obslh) :
            print('ObsLh file does not exist. Constructing observational likelihood...')
            likelihood(tag)
            print('#####################################')
            print('Observational likelihood complete!')
            print('#####################################')
            obslh_fname = 'ObsLh'+tagname+'.mat'
        else:
            print('Using existing observational likelihood file: ',obslh_fname)

        obslh_file=scipy.io.loadmat(obslh_fname, squeeze_me =False,struct_as_record=True)
        tide = obslh_file['tide'][0]
        # tide: activity level classification
        # 2 - low activity
        # 1 - moderate activity
        # 0 - high activity
        activity = ['HIGH', 'MODERATE', 'LOW']
        ObsLh = obslh_file['ObsLh']



        # determine days of iteration
        iters = np.shape(ObsLh)[0]
        # create particles and weights
        particles = np.empty([iters, N, 2])
        particles[0, :, :] = create_gaussian_particles(mean=(release_x, release_y), std=(50, 50), N=N)

        weights = np.zeros(N)

        total_weights = np.empty([iters, N])
        total_weights[0, :] = 0.0

        hdiff_coef = hdiff_coef_in_km2_per_day * 11.57

        # main loop 
        print('Particle filter geolocation for tag: '+tagname+'...')
        for x in range(1, iters):

            print('  Processing Day '+str(x+1)+'/'+str(iters)+'...')
            print('  Activity: '+activity[tide[x]]+', D = '+str(hdiff_coef[tide[x]])+' m^2/s')

            # Move: random walk substep, attreation term towards recap location
            predict(particles, hdiff=hdiff_coef[tide[x]], nsub = nsub, fvcom = fvcom, iterr = x)

            # Update: calculate weights
            weights = update(particles, weights, iterr = x, iObsLh = ObsLh[x, :], fvcom = fvcom)

            # plt.figure()
            # plt.scatter(particles[x,:,0], particles[x,:,1], c=weights)
            # plt.title('Before resample')

            # Resample: 
            if x < iters-1:
                indexes = systematic_resample(weights)
                weights = resample_from_index(particles, total_weights, weights, indexes)

            total_weights[x, :] = weights

            # total_weights[x, :] = weights

            # plt.figure()
            # plt.scatter(particles[x,:,0], particles[x,:,1], c=total_weights[x,:])
            # plt.title('After resample')

        # Most probable track (MPT): max total weight
        mpt_idx = np.argmax(total_weights.sum(axis=0))


        # save data
        result={'particles':particles, 'total_weights': total_weights, 'mpt_idx':mpt_idx}
        scipy.io.savemat('result'+tagname+'.mat',result)


        # plot data

        # for i in range(20):
        #     plt.plot(particles[i,:,0], particles[i,:,1],'.')


        # plot most probable track
        plt.figure()
        plt.plot(particles[:,mpt_idx,0], particles[:,mpt_idx,1])

        plt.plot(release_x, release_y, 'rx')
        plt.plot(recapture_x, recapture_y, 'r^')
        plt.savefig('track'+tagname+'.png')


if __name__ == '__main__':
   main()
