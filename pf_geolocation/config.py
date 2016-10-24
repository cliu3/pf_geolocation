path_to_tags = '/Users/cliu/Dropbox/Geolocation/projects/cod_zemeckis/tag_data/'

tagname_list = ['7_S11951']
tagid_list = [7]


N = 2000  # NUMBER OF PARTICLES

# Horizontal diffusivity coefficients fir high, moderate, and low activity days in km**2 per day:
hdiff_high = 100
hdiff_moderate = 50
hdiff_low = 10


nsub = 24  # NUMBER OF SUBSTEPS WITHIN A DAY

# path to FVCOM GOM mesh
fvcom_tidaldb = '/Users/cliu/Dropbox/Geolocation/preprocess/gen_tidal_db/fvcomdb_gom3_v2.mat'

# path to likelihood file:
lhpath = '/Users/cliu/Dropbox/Geolocation/Results/All_WGOM_cod/'