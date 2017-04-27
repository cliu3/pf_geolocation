import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
import sys

#tagname_list = ['7_S11951','8_S11938', '11_S11971', '12_S11974', '13_S11976', '16_S12060', '17_S12061', '18_S12059', '22_S12068', '24_S11845']
#tagid_list = [7,8,11,12,13,16,17,18,22,24]
tagid = int(sys.argv[1])

path_to_tags = '/home/cliu3/pf_geolocation/data/tag_files'
tag=scipy.io.loadmat(path_to_tags+'/'+str(tagid)+'_raw.mat',squeeze_me =False,struct_as_record=True)

#for tagname, tagid in zip(tagname_list,tagid_list):
    # plot raw tag data
#tag=scipy.io.loadmat(str(tagid)+'_raw.mat',squeeze_me =False,struct_as_record=True)
tag=tag['tag']
dnum=tag['dnum'][0,0][:,0]
temp=tag['temp'][0,0][:,0]
depth=tag['depth'][0,0][:,0]
dnum=dnum-678942

dnum_raw=tag['dnum_raw'][0,0][:,0]
depth_raw=tag['depth_raw'][0,0][:,0]
temp_raw=tag['temp_raw'][0,0][:,0]
dnum_raw=dnum_raw - 678942

tagname = str(tagid)+'_'+tag['tag_id'][0,0][0]

data=pd.DataFrame({'dnum':dnum,'temp':temp,'depth':depth})
data['DATE']=Time(data.dnum,format='mjd',scale='utc').datetime
data['DATE']=pd.to_datetime(data['DATE'])
#data['depth_demeaned']= data.depth-data.depth.mean()

#data_raw = pd.DataFrame({'dnum':dnum_raw,'temp':temp_raw,'depth':depth_raw})
#data_raw['DATE']=Time(data_raw.dnum,format='mjd',scale='utc').datetime
#data_raw['DATE']=pd.to_datetime(data_raw['DATE'])

# tbeg=np.floor(Time(data.loc[0].DATE,scale='utc').mjd)
# tend=np.ceil(Time(data.iloc[-1].DATE,scale='utc').mjd)

#plt.figure()
ax=data[['DATE','depth','temp']].plot(x='DATE',y=['depth','temp'],figsize=(9,4.5),secondary_y='temp',style=['b','r'],legend=False)
ax.set_ylabel('Depth (m)',fontsize=12)
ax.set_xlabel('')
ax.invert_yaxis()
ax.right_ax.set_ylabel('Temperature ($^\circ$C)',fontsize=12)
ax.set_title(tagname)


# plot shaded activity level
#lv=scipy.io.loadmat('/Users/cliu/Dropbox/Geolocation/smast_geolocate/run_dtcod_hpcc_cscvr/ObsLh'+tagname+'.mat',squeeze_me =False,struct_as_record=True)
lv=scipy.io.loadmat('/home/cliu3/pf_geolocation/data/likelihood_files/ObsLh'+tagname+'.mat',squeeze_me =False,struct_as_record=True)
tide = lv['tide'][0]

date_timestamp = data.DATE.dt.normalize().unique()

# loop over low, moderate activity levels
tide_list = [2,1]
alpha_list = [0.6,0.3]
print('Tag Name: '+tagname)
for itide,ialpha in zip(tide_list, alpha_list):
    list_date = [pd.Timestamp(i).date() for i in date_timestamp[tide==itide]]
    shade_index = [i.date() in list_date for i in data.DATE]
    ax.fill_between(data['DATE'].values, ax.get_ylim()[1], ax.get_ylim()[0], where=shade_index, color='green', alpha=ialpha,linewidth=0)
    if itide==2:
        print('Total low activity days: '+str(np.sum(tide==itide)))
    elif itide==1:
        print('Total moderate activity days: ' + str(np.sum(tide == itide)))



plotfile = 'tag_raw_activity' + tagname + '.pdf'
plt.savefig(plotfile, bbox_inches='tight')

plt.show()




