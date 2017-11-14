#path_to_tags = '/Users/cliu/Dropbox/Geolocation/projects/cod_zemeckis/tag_data/'
path_to_tags = 'data/tag_files'

#tagname_list = ['7_S11951']
tagid_list = [7] 

N = 10000  # NUMBER OF PARTICLES

# Horizontal diffusivity coefficients for high, moderate, and low activity days in km**2 per day:
hdiff_high = 10
hdiff_moderate = 5
hdiff_low = 1


nsub = 24  # NUMBER OF SUBSTEPS WITHIN A DAY

# path to FVCOM GOM mesh
#fvcom_tidaldb = '/Users/cliu/Dropbox/Geolocation/preprocess/gen_tidal_db/fvcomdb_gom3_v2.mat'
fvcom_tidaldb = 'data/fvcomdb_gom3_v2.mat'
fvcom_grid = 'data/gom3_grid_full.nc'
# path to FVCOM bottom temperature
bottom_temperature   = 'data/gom3_btemp_davged_20100501_20100610.nc'

# path to likelihood file:
use_existing_obslh = True 
lhpath = 'data/likelihood_files/'

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
