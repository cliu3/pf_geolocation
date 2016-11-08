from __future__ import print_function
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.driver as driver
import pycuda.curandom as curandom
from pycuda.compiler import SourceModule
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io
from pf_mod_gpu import *
from my_project import *
from filterpy.monte_carlo import systematic_resample
import os.path
from likelihood import *
from config import *




def main():
    """
    This is the main function of the particle filter geolocation package.
    """
    hdiff_coef_in_km2_per_day = np.array([hdiff_high, hdiff_moderate, hdiff_low])  # array of 3 [HIGH MODERATE LOW]

    # load FVCOM GOM mesh
    mat=scipy.io.loadmat(fvcom_tidaldb,squeeze_me=True, struct_as_record=False)
    fvcom=mat['fvcom']

    # initialize kernels
    global nearest
    global update_loc
    global resample_from_index_kernel
    mod_nearest = SourceModule(open('nearest.cu','r').read()) 
    nearest = mod_nearest.get_function('nearest')
    mod_update_loc = SourceModule(open('update_loc.cu','r').read())
    update_loc = mod_update_loc.get_function('update_loc')
    mod_resample_from_index_kernel = SourceModule(open('resample_from_index_kernel.cu','r').read())
    resample_from_index_kernel = mod_resample_from_index_kernel.get_function('resample_from_index_kernel')

    for (tagname, tagid) in zip(tagname_list, tagid_list):
        # load tag
        tag=scipy.io.loadmat(path_to_tags+str(tagid)+'_raw.mat',squeeze_me =False,struct_as_record=True)
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

        print('Processing tag: '+tagname+'...')

        #####################################
        # load observation likelihood data
        #####################################
        obslh_fname = lhpath+'ObsLh'+tagname+'.mat'
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

        #Define a pycuda based random number generator
        global rng
        rng = curandom.XORWOWRandomNumberGenerator() 

        # create particles and weights
        global particles
        global particle_x_gpu
        global particle_y_gpu
        
        particles = np.empty([iters, N, 2])
        
        particle_x_gpu, particle_y_gpu = create_gaussian_particles(mean=np.array([release_x, release_y]), std=np.array([50, 50]), N=N)
        particles[0, :, 0] = particle_x_gpu.get()
        particles[0, :, 1] = particle_y_gpu.get()

        weights = np.zeros(N)

        total_weights = np.empty([iters, N])
        total_weights[0, :] = 0.0

        hdiff_coef = hdiff_coef_in_km2_per_day * 11.57
        
        # initialize GPU arrays

        global xv_gpu
        global yv_gpu
        global nv_gpu
        global centers_gpu
        global nnodes
        global nelems
        global block        
        global grid

        xv = fvcom.x
        yv = fvcom.y
        nv = (fvcom.tri-1).T
        xc = fvcom.xc
        yc = fvcom.yc
        centers = (np.vstack([xc,yc]).T).flatten()
        
        nnodes = np.int32(fvcom.nverts)
        nelems = np.int32(fvcom.nelems)

        xv_gpu = gpuarray.to_gpu(xv.astype(np.float32))
        yv_gpu = gpuarray.to_gpu(yv.astype(np.float32))
        nv_gpu = gpuarray.to_gpu(nv.astype(np.uint32))
        centers_gpu = gpuarray.to_gpu(centers.astype(np.float32))

        # block grid sizes
        block_size = 512
        grid_size = int(math.ceil(N/block_size))
        block = (block_size,1,1)
        grid = (grid_size,1)



        # main loop 
        print('Particle filter geolocation for tag: '+tagname+'...')
        for x in range(1, iters):

            print('  Processing Day '+str(x+1)+'/'+str(iters)+'...')
            print('  Activity: '+activity[tide[x]]+', D = '+str(hdiff_coef[tide[x]])+' m^2/s')
            print('  # of particles: ', str(N))

            # Move: random walk substep, attreation term towards recap location
            predict(particles, hdiff=hdiff_coef[tide[x]], nsub = nsub, fvcom = fvcom, iterr = x)

            # Update: calculate weights
            weights = update(particles, weights, iterr = x, iObsLh = ObsLh[x, :], fvcom = fvcom)

            # plt.figure()
            # plt.scatter(particles[x,:,0], particles[x,:,1], c=weights)
            # plt.title('Before resample')

            # plt.figure()
            # plt.scatter(particle_x_gpu.get(), particle_y_gpu.get(), c=weights)
            # plt.title('Before resample (GPU)')

            # Resample: 
            if x < iters-1:
                indexes = systematic_resample(weights)
                weights = resample_from_index_gpu(particles, total_weights, weights, indexes)

            total_weights[x, :] = weights

            # total_weights[x, :] = weights

            # plt.figure()
            # plt.scatter(particles[x,:,0], particles[x,:,1], c=total_weights[x,:])
            # plt.title('After resample')

            # plt.figure()
            # plt.scatter(particle_x_gpu.get(), particle_y_gpu.get(), c=total_weights[x,:])
            # plt.title('After resample (GPU)')

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


def create_gaussian_particles(mean, std, N):
    # particles = np.empty((N, 2))
    # particles[:, 0] = mean[0] + (randn(N) * std[0])
    # particles[:, 1] = mean[1] + (randn(N) * std[1])

    global particles
    global rng

    ini_randx_gpu = rng.gen_normal(N,np.float32)
    ini_randy_gpu = rng.gen_normal(N,np.float32)
    particle_x_gpu =np.float32(mean[0])+ini_randx_gpu * np.float32(std[0])
    particle_y_gpu =np.float32(mean[1])+ini_randy_gpu * np.float32(std[1])

    particles[0, :, 0] = particle_x_gpu.get()
    particles[0, :, 1] = particle_y_gpu.get()

    # mean_gpu = gpuarray.to_gpu(mean)
    # std_gpu = gpuarray.to_gpu(std)
    # init_gaussian_kernel = ElementwiseKernel("float *x, float *rand, float *mean, float *std", "x[i] = mean + rand[i] * std", "init_gaussian_kernel")
    # init_gaussian_kernel(particle_x_gpu, randx_gpu, mean_gpu[0], std_gpu[0])
    # init_gaussian_kernel(particle_y_gpu, randy_gpu, mean_gpu[1], std_gpu[1])
    return particle_x_gpu, particle_y_gpu
    # driver.memcpy_dtod(particles_gpu[0,:].gpudata, randx_gpu.gpudata, particles_gpu[0,:].nbytes) #particles_gpu[0,:] = randx_gpu
    # driver.memcpy_dtod(particles_gpu[1,:].gpudata, randy_gpu.gpudata, particles_gpu[0,:].nbytes) #particles_gpu[1,:] = randy_gpu


    # return particles

def predict(particles,hdiff, iterr, nsub, fvcom):

    global rng
    global particle_x_gpu
    global particle_y_gpu
    global xv_gpu
    global yv_gpu
    global nv_gpu
    global centers_gpu
    global nnodes
    global nelems
    global block
    global grid
    global N
    global nearest
    global update_loc

    # xv = fvcom.x
    # yv = fvcom.y
    # nv = (fvcom.tri-1).T
    # xc = fvcom.xc
    # yc = fvcom.yc
    # centers = np.vstack([xc,yc]).T


    # stat = np.ones(N,dtype=np.int32)

    # x = particles[iterr-1, :, 0]
    # y = particles[iterr-1, :, 1]

    deltat = 86400.0 / nsub
    tscale = np.float32((2*deltat*hdiff)**0.5);


 

    for i in range(nsub):

        

        print("    Moving particles in substep "+str(i+1)+"/"+str(nsub)+"...")



        # horizontal random walk
        x0_gpu = gpuarray.empty_like(particle_x_gpu)
        y0_gpu = gpuarray.empty_like(particle_y_gpu)
        driver.memcpy_dtod(x0_gpu.gpudata, particle_x_gpu.gpudata, x0_gpu.nbytes)
        driver.memcpy_dtod(y0_gpu.gpudata, particle_y_gpu.gpudata, y0_gpu.nbytes)

        # x0_gpu = particle_x_gpu
        # y0_gpu = particle_x_gpu

        randx_gpu = rng.gen_normal(N,np.float32)
        randy_gpu = rng.gen_normal(N,np.float32)

        particle_x_gpu = particle_x_gpu + randx_gpu * tscale
        particle_y_gpu = particle_y_gpu + randy_gpu * tscale



        # x = x + (randn(N) * tscale)
        # y = y + (randn(N) * tscale)

        # find cell with nearest cell center
        minloc_gpu = gpuarray.empty(N,np.uint32)
        #mod_nearest = SourceModule(open('nearest.cu','r').read())
        #nearest = mod_nearest.get_function('nearest')
        nearest(particle_x_gpu, particle_y_gpu, np.int32(N), centers_gpu, nelems, minloc_gpu, block=block, grid=grid)
        # print(minloc_gpu[:5])

        # t_centers = cKDTree(centers)
        # pt = np.vstack([x,y]).T
        # _,minloc = t_centers.query(pt, k=1)


        # update locations for particles within FVCOM domain
        #mod_update_loc = SourceModule(open('update_loc.cu','r').read())
        #update_loc = mod_update_loc.get_function('update_loc')
        update_loc( particle_x_gpu, particle_y_gpu, x0_gpu, y0_gpu, xv_gpu, yv_gpu, nv_gpu, minloc_gpu, nelems, np.int32(N), block=block, grid=grid)

        # for i in range(N):
        #     # pt = np.vstack([x,y]).T[i]
        #     # _,minloc = t_centers.query(pt, k=1)
        #     # minval,minloc = nearest(centers,pt)
        #     # xtri = xv[nv[:,minloc]]
        #     # ytri = yv[nv[:,minloc]]
        #     # if isintriangle(xtri,ytri,pt[0],pt[1]):
        #     xtri = xv[nv[:,minloc[i]]]
        #     ytri = yv[nv[:,minloc[i]]]
        #     if isintriangle(xtri,ytri,pt[i,0],pt[i,1]):
        #         stat[i] = 1
        #     else:
        #         stat[i] = 0

        # # update particle state and reset particles on land 
        # x[stat==0] = x2[stat==0]
        # y[stat==0] = y2[stat==0]
        # randx_gpu.gpudata.free()
        # randy_gpu.gpudata.free()

    # dump end-of-day locations
    particles[iterr, :, 0] = particle_x_gpu.get()
    particles[iterr, :, 1] = particle_y_gpu.get()

def resample_from_index_gpu(particles, total_weights, weights, indexes):

    global particle_x_gpu
    global particle_y_gpu
    global block
    global grid

    particles[:,:,:] = particles[:,indexes,:]
    total_weights[:, :] = total_weights[:, indexes]
    weights = weights[indexes]
    weights /= np.sum(weights)

    # update index in GPU
    indexes_gpu = gpuarray.to_gpu(indexes.astype(np.uint32))
    # print(indexes[0:5])
    x0_gpu = gpuarray.empty_like(particle_x_gpu)
    y0_gpu = gpuarray.empty_like(particle_y_gpu)
    driver.memcpy_dtod(x0_gpu.gpudata, particle_x_gpu.gpudata, x0_gpu.nbytes)
    driver.memcpy_dtod(y0_gpu.gpudata, particle_y_gpu.gpudata, y0_gpu.nbytes)
    resample_from_index_kernel(particle_x_gpu, particle_y_gpu, x0_gpu, y0_gpu, indexes_gpu, np.int32(len(indexes)), block=block, grid=grid)

    return weights


if __name__ == '__main__':
   main()
