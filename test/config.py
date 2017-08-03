#path_to_tags = '/Users/cliu/Dropbox/Geolocation/projects/cod_zemeckis/tag_data/'
path_to_tags = '/home/cliu/pf_geolocation/data/tag_files'

#tagname_list = ['7_S11951']
tagid_list = [12] #56 days

N = 2000  # NUMBER OF PARTICLES

# Horizontal diffusivity coefficients for high, moderate, and low activity days in km**2 per day:
hdiff_high = 10
hdiff_moderate = 5
hdiff_low = 1


nsub = 24  # NUMBER OF SUBSTEPS WITHIN A DAY

# path to FVCOM GOM mesh
#fvcom_tidaldb = '/Users/cliu/Dropbox/Geolocation/preprocess/gen_tidal_db/fvcomdb_gom3_v2.mat'
fvcom_tidaldb = '/home/cliu/pf_geolocation/data/fvcomdb_gom3_v2.mat'
fvcom_grid = '/home/cliu/pf_geolocation/data/gom3_grid_full.nc'
# path to FVCOM bottom temperature
bottom_temperature   = '/home/cliu/pf_geolocation/data/gom3_btemp_davged_2003_2013.nc'

# path to likelihood file:
use_existing_obslh = True 
#lhpath = '/Users/cliu/Dropbox/Geolocation/Results/All_WGOM_cod/'
lhpath = '/home/cliu/pf_geolocation//data/likelihood_files/'

# tideLV: criteria for tidal signal detection in likelihood model
# tideLV  = [RMSE upper bound, R^2 lower bound, AMPLITUDE lower, AMPLITUDE upper]
tideLV  = [0.42, 0.85, 0.2, 2.0]

############################
# tag-specific paremeters  #
############################
std_temp_offset = 2.0 #higher value is more inclusive
tag_depth_range = 250 # in meters
tag_depth_accu = 0.008 # fraction of depth renge
tag_temp_accu = 0.1 # in degree C
