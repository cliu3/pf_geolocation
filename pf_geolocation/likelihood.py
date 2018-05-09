#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from scipy.stats import norm
import utils
from config import *

try: 
    xrange 
except NameError: 
    xrange = range

def likelihood(tag):
    """
    Construction of likelihood function after (Le Bris et al, 2013 eq (2))
    using daily max depth and depth where tidal signal is detected, with
    coorespoding temperature.

    Converted from Matlab code likelihood_cliu.m in hmm_smast
    """
    import netCDF4
    import scipy.io
    from my_project import my_project


    # loop over day to detect tidal signal and find daily best fit
    day_tidal_depth, day_tidal_depth_temp, day_max_depth, day_max_depth_temp = tidal_detection(tag)
    #######################
    # compute likelihood
    #######################
    # load variables
    int_dnum = np.floor(tag['dnum'])
    tbeg=int(np.floor(tag['dnum'].flatten()[0] -678942))
    tend=int(np.ceil(tag['dnum'].flatten()[-1] -678942))
    days = range(tbeg,tend)
    ndays = len(days)


    ## load FVCOM GOM mesh from tidal database
    print('Loading GOM FVCOM tidal database ... ')
    mat=scipy.io.loadmat(fvcom_tidaldb,squeeze_me=True, struct_as_record=False)
    fvcom=mat['fvcom']

    print('Calculating nodes surrounding each node ...')
    nbsn = nbsn_calc(fvcom)

    # compute depth S.D.
    #xt, yt = my_project(tag['release_lon'],tag['release_lat'],'forward')
    std_dep = std_dep_calc(fvcom, nbsn)

    # load FVCOM bottom temperature
    fin = netCDF4.Dataset(bottom_temperature)
    print('Loading temperature data ... ')
    # time
    time_mjd = fin.variables['time'][:]
    
    #####time_mdl = floor(time_mjd + datenum(1858,11,17,0,0,0))

    time_idx=np.flatnonzero( (time_mjd>=tbeg) & (time_mjd<=tend+1) )

    # bottom temperature
    
    t = fin.variables['temp'][time_idx, :].T
    #print('done loading temperature data')

    # loop over days
    ObsLh = np.empty([ndays,fvcom.nverts])*np.nan

    tide = np.zeros(ndays, dtype='int')
    # tide: activity level classification
    # 2 - low activity
    # 1 - moderate activity
    # 0 - high activity

    for i in xrange(ndays):
        print('day: '+str(i+1)+' of '+str(ndays))

        # calculate depth-based likelihood (ObsLh_dep_total)
        ObsLh_dep_total = ObsLh_dep_calc(i, std_dep, day_tidal_depth, day_max_depth, tide, fvcom)

        # calculate daily temperature S.D.
        std_temp = std_temp_calc(i, t, fvcom, nbsn)

        # calculate temp-based likelihood (ObsLh_temp_total)
        ObsLh_temp_total = ObsLh_temp_calc(i, t, std_temp, day_tidal_depth_temp, day_max_depth_temp, tide, fvcom.nverts)

        # Release/recapture locations treatment
        
        # ## release location treatment
        # xl, yl = my_project(tag['release_lon'],tag['release_lat'],'forward')
        # dist_rl = ( (fvcom.xc-xl)**2 + (fvcom.yc-yl)**2 )**0.5
        # rel_idx = np.argmin(dist_rl)
        # ObsLh[0,fvcom.tri[rel_idx,:]] = 1

        ## recapture location attraction likelihood
        if(tag['recap_uncertainty_km']>0):
            xr, yr = my_project(tag['recapture_lon'],tag['recapture_lat'],'forward')
            dist_r = ( (fvcom.x-xr)**2+(fvcom.y-yr)**2 )**0.5
            t_remain=ndays-i
            sigma = max( 1000*tag['recap_uncertainty_km'], 0.5*25000*t_remain)
            AttLh = norm.pdf(dist_r,0,sigma) #25000: typical cod swimming speed (30 cm/s)
            AttLh = AttLh/np.max(AttLh)
        else:
            AttLh = 1
        # calculate daily likelihood distribution
        ObsLh[i,:]=ObsLh_dep_total*ObsLh_temp_total*AttLh

    # TIDAL EXCLUSION based on RMSE
    print('Performing tidal-based exclusion to ObsLh using RMSE of tidal signal ... ')

    ## 13-hour fitting to determine eta_tag_fit
    eta_tag_fit, day_ampli, intv_cell = tidal_detection_long(tag, tide)

    ## reconstruction of FVCOM tidal signal using harmonics
    eta1, node_idx = FVCOM_predict(fvcom, tag['dnum'], day_ampli)

    ## Loop over days to calculate RMSE between tag nad FVCOM
    rmse_con = rmse_con_calc(ndays, tide, eta1, intv_cell, eta_tag_fit, fvcom.nverts, node_idx)
    
    ## update ObsLh
    #import ipdb; ipdb.set_trace()
    ObsLh = ObsLh * rmse_con



    tagname = str(np.asscalar(tag['fish_id']))+'_'+tag['tag_id'][0]
    scipy.io.savemat(lhpath+'/'+'ObsLh'+tagname+'.mat', {'ObsLh': ObsLh, 'tide': tide})
    #return ObsLh, tide

    


    

def tidal_detection(tag):
    """
    Detection of tidal signal from depth timeseries using a 5-hour moving 
    window, and return the daily max depth and tidal depth (if applicable)
    and the associated temperature.
    """
    #from config import *
    

    Twindow = 5  #time window = 5 h
    nwindow = int(np.asscalar(Twindow*3600//tag['min_intvl_seconds'])) # window size in data point numbers

    ntimes = len(tag['dnum'])

    int_dnum = np.floor(tag['dnum'])
    dbeg = int_dnum[0]
    dend = int_dnum[-1]
    days = range(int(dbeg), int(dend+1))
    ndays = len(days)

    
    #p: M2 period in hours
    p = 12.420601
    w=2*np.pi/(p/24) # Angular frequency
    sint = np.sin(w*tag['dnum'])
    cost = np.cos(w*tag['dnum'])



    # loop over day to detect tidal signal
    td_detected=np.empty(len(tag['dnum']))*np.nan
    td_used=td_detected.copy()
    day_tidal_depth=np.empty(ndays)*np.nan
    day_tidal_depth_temp=day_tidal_depth.copy()
    day_max_depth=np.empty(ndays)*np.nan
    day_max_depth_temp=np.empty(ndays)*np.nan
    print('Detecting tidal signal...')
    for i in range(ndays):
        print('day: '+str(i+1)+' of '+str(ndays))
        days_idx=np.where(int_dnum == days[i])[0]
        rmse=np.empty(len(days_idx))*np.nan
        rsquare=np.empty(len(days_idx))*np.nan
        ampli=np.empty(len(days_idx))*np.nan
        if (days_idx[0]+nwindow > ntimes):
            day_max_depth[i] = np.max(tag['depth'][days_idx])
            day_max_dep_ind = np.argmax(tag['depth'][days_idx])
            day_max_depth_temp[i]=tag['temp'][days_idx[day_max_dep_ind]]
            break
        
        day_max_depth[i]=np.max(tag['depth'][days_idx])
        day_max_dep_ind=np.argmax(tag['depth'][days_idx])
        day_max_depth_temp[i]=tag['temp'][days_idx[day_max_dep_ind]]
        # move window for each data point
        for j in range(len(days_idx)):
            if (days_idx[j]+nwindow > ntimes): break
            
            intv=range(days_idx[j], min(ntimes,days_idx[j]+nwindow-1) + 1 )
            rmse[j], rsquare[j], ampli[j] = lssinfit(np.ones(len(intv)), cost[intv], sint[intv], tag['depth'][intv])[0:3]

            
            crit = (rmse[j]<tideLV[0]) & (rsquare[j]>tideLV[1]) & (ampli[j]>tideLV[2]) & (ampli[j]<tideLV[3])
            if crit==1:
                td_detected[intv]=1
            
            
        
        
        # Find intervals with tidal information according to criteria
        crit = (rmse<tideLV[0]) & (rsquare>tideLV[1]) & (ampli>tideLV[2]) & (ampli<tideLV[3])

        # find best fit for each day and reconstruct corresponding fvcom signal
        if np.sum(crit)>0:
            idx=np.argwhere(rmse==np.min(rmse[crit])).item(0)
            
            intv=range(days_idx[idx], min(ntimes,days_idx[idx]+nwindow-1)+1 )
            td_used[intv]=1
            day_tidal_depth[i]=np.mean(tag['depth'][intv])
            day_tidal_depth_temp[i]=np.mean(tag['temp'][intv])


    return day_tidal_depth, day_tidal_depth_temp, day_max_depth, day_max_depth_temp

def lssinfit(ons,cost,sint,ts):
    """
    Fit a sinewave to input data by Least-Square

    Converted from Matlab (lssinfit.m from HMM geolocation toolbox)

    For more details see page 51 of Pedersen, M.W., 2007. Hidden Markov models 
    for geolocation of fish. Technical University of Denmark, DTU, DK-2800 Kgs. 
    Lyngby, Denmark.

    """

    

    out=0
    X=np.column_stack((ons, cost, sint))
    Y=ts 
    Y2=Y 
    n, m = np.shape(X) 
    # n is number of observations
    # m is number of paramters
    
    # Solve normal equations
    #theta = np.linalg.lstsq( (np.dot(np.transpose(X), X)) , np.dot(np.transpose(X), Y) )
    theta = np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X).dot(Y))
    
    Yhat1=np.dot(X, theta) # predictions
    res=Yhat1-Y # residuals
    
    rsquare = 1 - np.sum(res**2)/np.sum((Y-np.mean(Y))**2)
    rmse = np.sqrt(np.sum(res**2)/(n-m))
    ampli = np.sqrt(theta[1]**2 + theta[2]**2)
    lengthres = len(res)
    df = n-m-1
    S = np.sum(res**2)/(df)
    mwh = theta[0]
    alpha = theta[1]
    beta = theta[2]

    return rmse, rsquare, np.asscalar(ampli), out, Yhat1, mwh, alpha, beta

def nbsn_calc(fvcom):
    """
    determine nodes surrounding each node (no specific order)
    nbsn is padded with -999 as invalid value
    """
    
    # determine edges
    nEdges = fvcom.nelems*3
    tri = fvcom.tri-1
    edge = np.zeros([nEdges,2], dtype=int)
    icnt = 0
    for i in xrange(fvcom.nelems):
        #print(1, i)
        edge[icnt  ] = tri[i,[0,1]]   
        edge[icnt+1] = tri[i,[1,2]]   
        edge[icnt+2] = tri[i,[2,0]] 
        icnt = icnt + 3
    

    # determine nodes surrounding nodes (no specific order)
    ntsn = np.zeros([fvcom.nverts,1], dtype=int)-1
    nbsn = np.ones([fvcom.nverts,8], dtype=int)*-999

    for i in xrange(nEdges):
        #print(2, i)
        i1 = edge[i,0]
        i2 = edge[i,1]
        #lmin = np.min(np.abs(nbsn[i1,:]-i2))
        #if(lmin != 0):
        if i2 not in nbsn[i1,:]:
            ntsn[i1] = ntsn[i1]+1
            nbsn[i1,ntsn[i1]] = i2
        
        #lmin = np.min(np.abs(nbsn[i2,:]-i1))
        #if(lmin != 0):
        if i1 not in nbsn[i2,:]:
            ntsn[i2] = ntsn[i2]+1
            nbsn[i2,ntsn[i2]] = i1

    return nbsn
        
def std_dep_calc(fvcom, nbsn):
    print('Calculating depth S.D. ...')
    std_dep=np.empty(fvcom.nverts)*np.nan
    nnodes = fvcom.nverts
    for nd in xrange(nnodes):
        if(nd%3000 == 0) or (nd == nnodes-1): print('  ', int((float(nd+1)/nnodes)*100),"%")
        
        # progress output
        # if (mod(nd,500)==0):
        #     fprintf('node: %d/%d\n',nd,fvcom.nverts)
        
        nnode_list=nbsn[nd,:]
        nnode_list=nnode_list[nnode_list >= 0]
        std_dep[nd]=np.std(fvcom.dep[nnode_list]-fvcom.dep[nd], ddof=1)


    return std_dep
     
def ObsLh_dep_calc(i, std_dep, day_tidal_depth, day_max_depth, tide, fvcom):
    print('  Calculating depth-based likelihood ...')

    if 'std_depth_offset' not in locals():
        std_depth_offset=2.0 #higher value is more inclusive
    
    std_dep = std_dep + std_depth_offset
    if 'tag_depth_range' not in locals():
        tag_depth_range = 250 # in meter
    
    if 'tag_depth_accu' not in locals():
        tag_depth_accu = 0.008 # fraction of depth renge
    

    if np.isfinite(day_tidal_depth[i]):
        tide[i]=1
        ObsLh_dep_tidal = norm.cdf((day_tidal_depth[i]+tag_depth_range*tag_depth_accu)*np.ones(fvcom.nverts),fvcom.dep,std_dep) - \
            norm.cdf((day_tidal_depth[i]-tag_depth_range*tag_depth_accu)*np.ones(fvcom.nverts),fvcom.dep,std_dep)
        ObsLh_dep_tidal = ObsLh_dep_tidal / np.max(ObsLh_dep_tidal)
        
        ObsLh_dep_total=ObsLh_dep_tidal
    else:
        tide[i]=0
        ObsLh_dep = norm.cdf( -day_max_depth[i]*np.ones(fvcom.nverts), -fvcom.dep,std_dep) / \
            norm.cdf(np.zeros(fvcom.nverts),-fvcom.dep,std_dep)
        ObsLh_dep = ObsLh_dep / np.max(ObsLh_dep)

        ObsLh_dep_total=ObsLh_dep


    return ObsLh_dep_total

def std_temp_calc(i, t, fvcom, nbsn):
    # compute temp std for neighboring nodes
    if 'std_temp_offset' not in locals():
        std_temp_offset=2.0 #higher value is more inclusive

    std_temp=np.empty(fvcom.nverts)*np.nan
    nnodes = fvcom.nverts
    print('  Calculating temperature S.D. ...')
    #[~,iframe] = min(abs(int_dnum(i)-time_mdl))
    for nd in xrange(nnodes):
        
        
        nnode_list=nbsn[nd,:]
        nnode_list=nnode_list[nnode_list >= 0]
        std_temp[nd]=np.std(t[nnode_list,i]-t[nd,i], ddof=1)
        
        
    
    std_temp=std_temp+std_temp_offset


    return std_temp

def ObsLh_temp_calc(i, t, std_temp, day_tidal_depth_temp, day_max_depth_temp, tide, nverts):
    print('  Calculating temperature-based likelihood ...')
    if 'tag_temp_accu' not in locals():
        tag_temp_accu = 0.1 # in degree C
    #if np.isfinite(day_tidal_depth[i]):
    if tide[i]==1:
        ObsLh_temp_tidal = norm.cdf((day_tidal_depth_temp[i]+tag_temp_accu)*np.ones(nverts),t[:,i],std_temp)- \
            norm.cdf((day_tidal_depth_temp[i]-tag_temp_accu)*np.ones(nverts),t[:,i],std_temp)
        ObsLh_temp_tidal = ObsLh_temp_tidal / np.max(ObsLh_temp_tidal)
        ObsLh_temp_total=ObsLh_temp_tidal
    else:
        ObsLh_temp = norm.cdf((day_max_depth_temp[i]+tag_temp_accu)*np.ones(nverts),t[:,i],std_temp)- \
            norm.cdf((day_max_depth_temp[i]-tag_temp_accu)*np.ones(nverts),t[:,i],std_temp)
        ObsLh_temp = ObsLh_temp / np.max(ObsLh_temp)
        ObsLh_temp_total=ObsLh_temp


    return ObsLh_temp_total

def tidal_detection_long(tag, tide):

    Twindow = 13  #time window = 13 h
    nwindow = int(np.asscalar(Twindow*3600//tag['min_intvl_seconds'])) # window size in data point numbers

    ntimes = len(tag['dnum'])

    int_dnum = np.floor(tag['dnum'])
    dbeg = int_dnum[0]
    dend = int_dnum[-1]
    days = range(int(dbeg), int(dend+1))
    ndays = len(days)

    #p: M2 period in hours
    p = 12.420601
    w=2*np.pi/(p/24) # Angular frequency
    sint = np.sin(w*tag['dnum'])
    cost = np.cos(w*tag['dnum'])

    day_ampli=np.empty(ndays)*np.nan
    intv_cell = [None]*ndays
    eta_tag_fit = [None]*ndays
    print('Detecting long tidal signal...')
    for i in range(ndays):
        print('  day: '+str(i+1)+' of '+str(ndays))
        days_idx=np.where(int_dnum == days[i])[0]
        rmse=np.empty(len(days_idx))*np.nan
        rsquare=np.empty(len(days_idx))*np.nan
        ampli=np.empty(len(days_idx))*np.nan
        if (days_idx[0]+nwindow > ntimes):
            break

        #move window for each data point
        for j in range(len(days_idx)):
            if (days_idx[j]+nwindow > ntimes): break
            
            intv=range(days_idx[j], min(ntimes,days_idx[j]+nwindow-1) + 1 )
            rmse[j], rsquare[j], ampli[j] = lssinfit(np.ones(len(intv)), cost[intv], sint[intv], tag['depth'][intv])[0:3]
            
        
        
        # Find intervals with tidal information according to criteria
        crit = (rmse<tideLV[0]) & (rsquare>tideLV[1]) & (ampli>tideLV[2]) & (ampli<tideLV[3])

        #find best fit for each day and reconstruct corresponding fvcom signal
        if np.sum(crit)>0:
            tide[i] = 2
            
            idx=np.argwhere(rmse==np.min(rmse[crit])).item(0)
            intv=range(days_idx[idx], min(ntimes,days_idx[idx]+nwindow-1)+1 )
            intv_cell[i]=intv
            time=tag['dnum'][intv].T[0]
            day_ampli[i] = ampli[idx]
            
            eta_tag = ( tag['depth'][intv]-np.mean(tag['depth'][intv]) ).T[0]
            
            # nonlinear sine fit
            eta_tag_fit[i] = utils.fit(time,eta_tag)
            
            
    return eta_tag_fit, day_ampli, intv_cell

def FVCOM_predict(fvcom, dnum, day_ampli):
    """
    Reconstruction of FVCOM tidal signal
    """
    from astropy.time import Time

    # #Define tidal constituents and values
    # inpcon = fvcom.comps

    # only consider nodes whose range of amplitude falls in range of amplitude
    # of fitted signal +/- ampl_buffer (in meters)
    ampl_buffer = 0.0
    Fr1=0.01*( fvcom.amp[:,0] - np.sum(fvcom.amp[:,1:],1) )
    Fr1[Fr1<=0] = 0
    Fr2=0.01*np.sum(fvcom.amp,1)
    Tr1=max(0,min(day_ampli)-ampl_buffer)
    Tr2=max(day_ampli)+ampl_buffer
    node_idx = np.where( (Fr2>=Tr1) & (Fr1<=Tr2) )[0]

    pha1=np.zeros( (len(node_idx),len(fvcom.comps)) )
    amp1=np.zeros( (len(node_idx),len(fvcom.comps)) )
    print('Reconstructing FVCOM tidal signal ... ')

    tag_time = Time(dnum-678942,format='mjd',scale='utc').datetime[:,0]
    eta1 = [None]*len(node_idx)
    pha1 = fvcom.phase[node_idx,:]
    amp1 = fvcom.amp[node_idx,:]*0.01 #cm to m    
    for nd in xrange(len(node_idx)):
        # if(nd%3000 == 0) or (nd == nnodes-1): print( int((float(nd+1)/nnodes)*100),"%")
        if (nd+1)%500 == 0:
            print('  node: '+str(nd+1)+'/'+str(len(node_idx))+' -- ', int((float(nd+1)/len(node_idx))*100),"%")

        # create timeseries with the defined tidal harmonics info with ttide
        eta1[nd]= utils.predict(tag_time, amp1[nd], pha1[nd])
        
    return eta1, node_idx


def rmse_con_calc(ndays, tide, eta1, intv_cell, eta_tag_fit, nverts, node_idx):

    print('Applying threashold for tidal fit ... ')
    rmse_con = np.ones((ndays,nverts))
    for i in xrange(ndays):
        print('  Processing day: ', str(i+1), ' of ', str(ndays), '...')

        # calculate rmse map
        if (tide[i]==2):
            rmse_eta = np.empty(nverts)*np.nan
            
            intv = intv_cell[i]
            
            for nd in xrange(len(node_idx)):
                
                #calculate tidal signal RMSE
                eta1_window=eta1[nd][intv]-np.mean(eta1[nd][intv])
                rmse_eta[node_idx[nd]]= np.sqrt(np.mean( (eta_tag_fit[i]-eta1_window)**2))
                
                
            
            thresh = np.nanmin(rmse_eta)+0.3*(np.nanmax(rmse_eta) - np.nanmin(rmse_eta))
            rmse_con[i,:]=0
            rmse_con[i,rmse_eta<=thresh]=1

    return rmse_con
