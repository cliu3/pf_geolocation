from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import scipy.io
import scipy.stats
from astropy.time import Time
import pandas as pd
from my_project import *
import os.path
#from config import *
import sys


#tagid = 7
# try:
tagid = int(sys.argv[1])
#tagid = 12
# except:


# load tag file
path_to_tags = '/home/cliu3/pf_geolocation/data/tag_files'
tag=scipy.io.loadmat(path_to_tags+'/'+str(tagid)+'_raw.mat',squeeze_me =False,struct_as_record=True)
tag=tag['tag'][0,0]
release_lon = tag['release_lon'][0,0]
release_lat = tag['release_lat'][0,0]
[release_x, release_y] = my_project(release_lon, release_lat, 'forward')
recapture_lon = tag['recapture_lon'][0,0]
recapture_lat = tag['recapture_lat'][0,0]
[recapture_x, recapture_y] = my_project(recapture_lon, recapture_lat, 'forward')

tagname = str(tagid)+'_'+tag['tag_id'][0]

# load result file
result = scipy.io.loadmat('result'+tagname+'.mat',squeeze_me =False,struct_as_record=True)
particles = result['particles']
mpt_idx =  result['mpt_idx']
# determine most probable track
mpt_x = particles[:,mpt_idx,0].flatten()
mpt_y = particles[:,mpt_idx,1].flatten()
(mpt_lon, mpt_lat) = my_project(mpt_x, mpt_y, 'reverse')

day_dnum = np.array(range(int(tag['dnum'][0]), int(tag['dnum'][-1])+1))
date = Time(day_dnum-678942,format='mjd',scale='utc').datetime
MPT = pd.DataFrame({'date':date, 'lon':mpt_lon, 'lat':mpt_lat, 'X':mpt_x, 'Y':mpt_y})
MPT['date'] = pd.to_datetime(MPT['date'])
MPT = MPT[['date', 'X', 'Y', 'lat', 'lon']]
MPT.to_csv('mpt_'+tagname+'.csv')
#-- calculate cumulative probability distribution
# construct daily distrubution using kernel density estimation
xmin = particles[:,:,0].min()
xmax = particles[:,:,0].max()
ymin = particles[:,:,1].min()
ymax = particles[:,:,1].max()
X, Y = np.meshgrid(np.linspace(xmin,xmax,50), np.linspace(ymin,ymax,50))
positions = np.vstack([X.ravel(), Y.ravel()])

ndays = len(particles)
udist = np.zeros_like(X)

# for i in range(ndays):
#     print("Processing kde for Day "+str(i+1)+"/"+str(ndays)+"...")
#     values = particles[i].T
#     kernel = scipy.stats.gaussian_kde(values)
#     Z = np.reshape(kernel(positions).T, X.shape)
#     Z = Z/Z.max()
#     udist += Z
print("Processing kde...")
values = np.vstack([particles[:,:,0].flatten(), particles[:,:,1].flatten()])
kernel = scipy.stats.gaussian_kde(values)
udist = np.reshape(kernel(positions).T, X.shape)
udist = udist/udist.max()

scipy.io.savemat('UD_'+tagname+'.mat',{'X':X, 'Y':Y, 'udist':udist})

# create basemap
print('Generating plot...')
latStart = 41.15
latEnd   = 43.15
lonStart =-71
lonEnd   =-68

map = Basemap(projection='merc', lat_0 = 42, lon_0 = -70,resolution = 'h', area_thresh = 0.1,llcrnrlon=lonStart, llcrnrlat=latStart,
    urcrnrlon=lonEnd, urcrnrlat=latEnd)
map.fillcontinents(color = 'green')

#-- plot mpt
mptlon, mptlat = my_project(mpt_x, mpt_y, 'inverse')
mptx, mpty = map(mptlon, mptlat)
map.plot(mptx,mpty,'b-')
#plot release and recapture location

map.plot(mptx[0],mpty[0],'kx',label="Release")
recap_x, recap_y = map(recapture_lon, recapture_lat)
map.plot(recap_x, recap_y,'k^',markeredgecolor='k',label="Reported Recapture")
map.plot(mptx[-1],mpty[-1],'bv',markeredgecolor='b',label="Simulated Recapture")


#-- plot uncertainty distribution
lon_g, lat_g = my_project(X, Y, 'inverse')
map.pcolormesh(lon_g, lat_g,udist,cmap=plt.cm.cubehelix_r,latlon=True,shading='gouraud')

plt.legend(numpoints=1,prop={'size':16},loc='lower right')
plt.title(tagname+' gpu')

plt.savefig('track'+tagname+'_gpu.pdf', dpi=300, bbox_inches='tight')
